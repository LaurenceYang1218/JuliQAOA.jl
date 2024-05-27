using JuliQAOA, Graphs
using Optim, LineSearches
using Statistics
using Random
using Enzyme

using CUDA
using CUDA: i32

Random.seed!(1)
n = 10
pround = 5

angle = rand(2*pround) .* 2π
graph = erdos_renyi(n, 0.5)

mixer = mixer_x(n)
obj_vals = [maxcut(graph, x) for x in states(n)]

println("n: $(n), pround: $(pround)")

function initH(mixer::Mixer{X})
    sv = ones(ComplexF64, mixer.N)/sqrt(mixer.N)
    return sv
end

function RZZGate!(d_sv, d_angles, d_objvals, pidx::Int64)
    tidx = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    rzz_angle = d_angles[pidx] * d_objvals[tidx] 

    rzz_complex = cos(rzz_angle) + sin(rzz_angle)im

    in = d_sv[tidx]
    real_sv = real(in) * real(rzz_complex) + imag(in) * imag(rzz_complex)
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


f(x) = -exp_value(x, mixer, obj_vals)/maximum(obj_vals)
f2(x) = -circuit(n, pround, x, mixer, obj_vals)/maximum(obj_vals)

println("Approximation ratio circuit: $(circuit(n, pround, angle, mixer, obj_vals)/maximum(obj_vals))")
println("Approximation ratio exp_value: $(exp_value(angle, mixer, obj_vals)/maximum(obj_vals))")

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

approx = circuit(n, pround, optim_angle, mixer, obj_vals)/maximum(obj_vals)
println("Approximation ratio circuit (optimized): $(approx)")

approx = exp_value(optim_angle, mixer, obj_vals)/maximum(obj_vals)
println("Approximation ratio exp_value (optimized): $(approx)")

