module FDDerivatives

function makejacobian(f::Function, h=sqrt(eps(Float64)))
	function jacobian(x::Vector)
		xphs = Array(Array{Float64, 1}, length(x) + 1)
		for i = 1:length(x)
			xphs[i] = copy(x)
			xphs[i][i] += h
		end
		xphs[end] = copy(x)
		ys = pmap(f, xphs)
		J = Array(eltype(ys[1]), length(ys[1]), length(x))
		for i = 1:length(x)
			J[:, i] = ys[i] - ys[end]
		end
		scale!(J, 1 / h)
		return J
	end
end

end
