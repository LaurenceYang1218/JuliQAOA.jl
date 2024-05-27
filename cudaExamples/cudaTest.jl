using JuliQAOA, Graphs
using Optim, LineSearches
using Statistics
using Random
using Enzyme

using CUDA
using CUDA: i32

Random.seed!(1)
n = 6
pround = 1

angle = rand(2*pround) .* 2π
graph = erdos_renyi(n, 0.5)

mixer = mixer_x(n)
obj_vals = [maxcut(graph, x) for x in states(n)]

println("n: $(n), pround: $(pround)")

"""
Apply the Hadamard gate ``H^{\\otimes n} to all qubits in `sv`, modifying `sv` in-place.
"""
function applyH!(sv)
    n = Int(log2(length(sv)))
    @inbounds for i in 1:n
        applyH!(sv,i)
    end
end

"""
Apply the Hadamard gate ``H^{\\otimes n} to the `l`th qubit in `sv`, modifying `sv` 
in-place.

Code modified (with permision) from https://blog.rogerluo.dev/2020/03/31/yany/.
"""
function applyH!(sv, l)
    U11 = 1/sqrt(2); U12 = 1/sqrt(2);
    U21 = 1/sqrt(2); U22 = -1/sqrt(2);
    step_1 = 1 << (l - 1)
    step_2 = 1 << l

    @inbounds if step_1 == 1
        for j in 0:step_2:size(sv, 1)-step_1
            ST1 = U11 * sv[j + 1] + U12 * sv[j + 1 + step_1]
            ST2 = U21 * sv[j + 1] + U22 * sv[j + 1 + step_1]

            sv[j + 1] = ST1
            sv[j + 1 + step_1] = ST2
        end
    elseif step_1 == 2
        for j in 0:step_2:size(sv, 1)-step_1
            Base.Cartesian.@nexprs 2 i->begin
                ST1 = U11 * sv[j + i] + U12 * sv[j + i + step_1]
                ST2 = U21 * sv[j + i] + U22 * sv[j + i + step_1]
                sv[j + i] = ST1
                sv[j + i + step_1] = ST2    
            end
        end
    elseif step_1 == 4
        for j in 0:step_2:size(sv, 1)-step_1
            Base.Cartesian.@nexprs 4 i->begin
                ST1 = U11 * sv[j + i] + U12 * sv[j + i + step_1]
                ST2 = U21 * sv[j + i] + U22 * sv[j + i + step_1]
                sv[j + i] = ST1
                sv[j + i + step_1] = ST2    
            end
        end
    elseif step_1 == 8
        for j in 0:step_2:size(sv, 1)-step_1
            Base.Cartesian.@nexprs 8 i->begin
                ST1 = U11 * sv[j + i] + U12 * sv[j + i + step_1]
                ST2 = U21 * sv[j + i] + U22 * sv[j + i + step_1]
                sv[j + i] = ST1
                sv[j + i + step_1] = ST2    
            end
        end
    else
        for j in 0:step_2:size(sv, 1)-step_1
            for i in j:8:j+step_1-1
                Base.Cartesian.@nexprs 8 k->begin
                    ST1 = U11 * sv[i + k] + U12 * sv[i + step_1 + k]
                    ST2 = U21 * sv[i + k] + U22 * sv[i + step_1 + k]
                    sv[i + k] = ST1
                    sv[i + step_1 + k] = ST2
                end
            end
        end
    end
end


function initH(mixer::Mixer{X})
    sv = ones(ComplexF64, mixer.N)/sqrt(mixer.N)
    return sv
end

function RZZGate!(d_sv, d_angles, d_objvals, pidx::Int64)
    tidx = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    rzz_angle = d_angles[pidx] * d_objvals[tidx] 

    rzz_complex = cos(rzz_angle) + sin(rzz_angle)im

    in = d_sv[tidx]
    real_sv = real(in) * real(rzz_complex) - imag(in) * imag(rzz_complex)
    imag_sv = -real(in) * imag(rzz_complex) + imag(in) * real(rzz_complex)
    d_sv[tidx] = real_sv + (imag_sv)im
    
    return  
end

function RZZGate(d_sv, d_angles, d_objvals, 
                    p::Int64, pround::Int64, numStates::Int64) 
    if numStates > 128
        blockNum = Int64(numStates/128)
        threadNum = 128
    else 
        blockNum = 1
        threadNum = numStates
    end
    @cuda blocks=blockNum threads=threadNum RZZGate!(d_sv, d_angles, d_objvals, p+pround)
    return
end

function RXGate!(d_sv, d_angles, d_mixer, targetQubit::Int64, pidx::Int64, numQubits::Int64)
    tidx = (blockIdx().x-1i32) * blockDim().x + threadIdx().x

    mask = (1 << targetQubit) - 1;
    up_off = (((tidx-1) >> targetQubit) << (targetQubit + 1)) | ((tidx-1) & mask)
    lo_off = up_off + (1 << targetQubit)
    up_off = up_off + 1;
    lo_off = lo_off + 1;

    up_rx_angle = (d_angles[pidx]*d_mixer[up_off])/numQubits
    lo_rx_angle = (d_angles[pidx]*d_mixer[lo_off])/numQubits
    up_rx_complex = cos(up_rx_angle) + sin(up_rx_angle)im
    lo_rx_complex = cos(lo_rx_angle) + sin(lo_rx_angle)im
    
    # RX-gate
    up = d_sv[up_off]
    lo = d_sv[lo_off]

    real_sv_up = real(up) * real(up_rx_complex) + imag(up) * imag(up_rx_complex)
    imag_sv_up = imag(up) * real(up_rx_complex) - real(up) * imag(up_rx_complex)
    
    real_sv_lo = imag(lo) * imag(lo_rx_complex) + real(lo) * real(lo_rx_complex)
    imag_sv_lo = -real(lo) * imag(lo_rx_complex) + imag(lo) * real(lo_rx_complex)
    
    d_sv[up_off] = real_sv_up + (imag_sv_up)im
    d_sv[lo_off] = real_sv_lo + (imag_sv_lo)im 

    return
end

function RXGate(d_sv, d_angles, d_mixer, targetQubit::Int64,
                    p::Int64, numQubits::Int64, numStates::Int64)
    if (numStates >> 1) > 128
        blockNum = Int64((numStates >> 1)/128)
        threadNum = 128
    else 
        blockNum = 1
        threadNum = numStates >> 1
    end
    
    @cuda blocks=blockNum threads=threadNum RXGate!(d_sv, d_angles, d_mixer,
                                                targetQubit, p, numQubits)
    return
end

function HGate!(d_sv, targetQubit::Int64)
    tidx = (blockIdx().x-1i32) * blockDim().x + threadIdx().x

    mask = (1 << targetQubit) - 1;
    up_off = (((tidx-1) >> targetQubit) << (targetQubit + 1)) | ((tidx-1) & mask)
    lo_off = up_off + (1 << targetQubit)
    up_off = up_off + 1;
    lo_off = lo_off + 1;

    h_angle = 1 / sqrt(2)
    # H-gate
    up = d_sv[up_off]
    lo = d_sv[lo_off]
    
    real_sv_up = (real(up) + real(lo)) * h_angle
    imag_sv_up = (imag(up) + imag(lo)) * h_angle
    real_sv_lo = (real(up) - real(lo)) * h_angle
    imag_sv_lo = (imag(up) - imag(lo)) * h_angle

    d_sv[up_off] = real_sv_up + (imag_sv_up)im
    d_sv[lo_off] = real_sv_lo + (imag_sv_lo)im
    
    return
end

function HGate(d_sv, targetQubit::Int64, numStates::Int64)
    if (numStates >> 1) > 128
        blockNum = Int64((numStates >> 1)/128)
        threadNum = 128
    else    
        blockNum = 1
        threadNum = numStates >> 1
    end
    
    @cuda blocks=blockNum threads=threadNum HGate!(d_sv, targetQubit)
    
    return 
end

function circuit_kernel(n::Int64, pround::Int64, d_sv, d_angles, d_mixer, d_objvals, N)

    for p in 1:pround
        RZZGate(d_sv, d_angles, d_objvals, p, pround, N)

        for i in 0:n-1
            HGate(d_sv, i, N)
        end
        for i in 0:n-1
            RXGate(d_sv, d_angles, d_mixer, i, p, n, N)
        end
        for i in 0:n-1
            HGate(d_sv, i, N)
        end
    end
    return 
end

function circuit!(n::Int64, pround::Int64, sv, angles, mixer, obj_vals, N)

    d_sv = CuArray(sv)
    d_angles = CuArray(angles)
    d_mixer = CuArray(mixer)
    d_objvals = CuArray(obj_vals)
    circuit_kernel(n, pround, d_sv, d_angles, d_mixer, d_objvals, N)
    sv = Array(d_sv)

    # println(sv)

    for i in eachindex(sv)
        sv[i] = abs2(sv[i])
        sv[i] *= obj_vals[i]
    end
    
    return real(sum(sv))

end

function circuit(n::Int64, pround::Int64, angles::Vector{Float64}, 
                            mixer::Mixer{X}, obj_vals::Vector{Int64})
    sv = initH(mixer)
    return circuit!(n, pround, sv, angles, mixer.d, obj_vals, mixer.N)
end

function circuit_gt!(n::Int64, pround::Int64, sv, angles, mixer, obj_vals, N)
    for i in 1:pround
        @inbounds for j in eachindex(sv)
            sv[j] *= exp(-1im*angles[i+pround]*obj_vals[j])
        end
        applyH!(sv)
        @inbounds for j in eachindex(sv)
            sv[j] *= exp(-1im*angles[i]*mixer[j])
        end
        applyH!(sv)
    end

    # println(sv)

    for i in eachindex(sv)
        sv[i] = abs2(sv[i])
        sv[i] *= obj_vals[i]
    end

    return real(sum(sv))
end

function circuit_gt(n::Int64, pround::Int64, angles::Vector{Float64}, 
                            mixer::Mixer{X}, obj_vals::Vector{Int64})
    sv = initH(mixer)
    return circuit_gt!(n, pround, sv, angles, mixer.d, obj_vals, mixer.N)
end

f(x) = -exp_value(x, mixer, obj_vals)/maximum(obj_vals)
f2(x) = -circuit(n, pround, x, mixer, obj_vals)/maximum(obj_vals)

println("circuit: $(circuit(n, pround, angle, mixer, obj_vals)/maximum(obj_vals))")
println("circuit_gt: $(circuit_gt(n, pround,angle, mixer, obj_vals)/maximum(obj_vals))")
println("exp_value: $(exp_value(angle, mixer, obj_vals)/maximum(obj_vals))")



function g!(G, x)
    G .= 0
    tmpG = similar(G)
    # calculate the gradient for the specific graph and store in tmpG
    grad!(tmpG, x, mixer, obj_vals; flip_sign=true) 
    # add the results to the overall gradient G
    # remembering to divide by the maximum value to get the approximation ratio
    G .+= tmpG/maximum(obj_vals)
end

minimizer = optimize(f, g!, angle, BFGS(linesearch=LineSearches.BackTracking()))
optim_angle = Optim.minimizer(minimizer)

# println("angle: $(angle)")
# println("optim_angle: $(optim_angle)")

approx = circuit(n, pround, optim_angle, mixer, obj_vals)/maximum(obj_vals)
println("Approximation ratio circuit: $(approx)")

approx = exp_value(optim_angle, mixer, obj_vals)/maximum(obj_vals)
println("Approximation ratio exp_value: $(approx)")








# function grad!(G::Vector, angles::Vector, mixer::Mixer, obj_vals::AbstractVector, measure::AbstractVector=obj_vals; flip_sign=false)
#     sv = initH(mixer)
#     grad!(G, sv, angles, mixer, obj_vals, measure; flip_sign=flip_sign)
# end

# function grad!(G::Vector, sv::Vector, angles::Vector, mixer::Mixer, obj_vals::AbstractVector, measure::AbstractVector=obj_vals; flip_sign=false)
#     sv_copy = copy(sv)
#     sv_gpu = CuArray(sv_copy)
#     dsv = zeros(ComplexF64, mixer.N)
#     dsv_gpu = CuArray(dsv)
#     angles_gpu = CuArray(angles)
#     G .= 0.0
#     G_gpu = CuArray(G)
#     f(a,b) = (flip_sign ? -1 : 1)*circuit_kernel(n, pround, a, b, mixer.d, obj_vals, mixer.N)  # 
#     @cuda threads=length(sv_copy) Enzyme.autodiff_deferred(Reverse, f, Duplicated(sv_gpu, dsv_gpu), Duplicated(angles_gpu, G_gpu))
# end