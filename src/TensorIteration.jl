"""
	TensorIteration

Macros to iterate over multiple arrays with a common set of indices, i.e. create loops
for custom tensor operations. The macro
```
@tensoriter ax[Adims => A, Bdims => B,...] begin
      (body)
   end
```
iterates over arrays `A`,`B`,..., where `axes(A) = ax[Adims]`, etc., and within `body`, elements of `A` are denoted by `A[]`, etc. Specifically, the macro creates the loop
```
for I in CartesianIndices(ax)
   body′
end
```
where `body′` is obtained from `body` by replacing each occurrence of
`A[]` by `A[I[Adims]]`, each occurrence of `B[]` by `B[I[Bdims]]`, etc.

`@stridediter` is equivalent to `@tensoriter` but specialized for strided arrays with
linear indexing, resulting in slightly better performance.

# Examples
```
# Matrix multiplication
C = zeros(axes(A,1), axes(B,2))
ax = (axes(A,1), axes(A,2), axes(B,2))
@stridediter ax[(1,2) => A, (2,3) => B, (1,3) => C] begin
   C[] += A[] * B[]
end

# trace
s = 0.0
@stridediter axes(A,1)[(1,1) => A] begin
   s += A[]
end
```
"""
module TensorIteration

using SuperTuples: accumtuple
using MacroTools
using MacroTools: postwalk

# """
# Syntax:
#
# @miter for (iA, iB,...) in Axes[(5,1,6), (3,2)], with(A,B,k)
# 	C[iC] = A[iA] + B[iB] + (k += 1.0)
# end
# """


export @stridediter


const Axes{N} = NTuple{N, AbstractUnitRange{<:Integer}}

"""
```
@stridediter ax[Adims => A, Bdims => B,...] begin
      (body)
   end
```
Iterate over strided arrays `A`,`B`,..., where `axes(A) = ax[Adims]`, etc. See [`TensorIteration`](@ref) for details.
"""
macro stridediter(header, expr)
	@capture(expr, begin body_ end) || error("invalid syntax")

	# Parse header
	@capture(header, ax_[maps__]) || error("Invalid syntax")
	ax = esc(ax)

	loopvar = esc(gensym("cindex"))
	sizevar = esc(gensym("axsize"))

	iterdefs = []
	stride_calcs = []
	index_asgns = []
	ax_checks = []
	for amap in maps
		# parse array arguments and their dimensions
		@capture(amap, adims_ => arr_)
		idx = gensym("i")
		push!(iterdefs, (arr, adims, idx))
		arr = esc(arr)
		adims = esc(adims)

		# code that checks to make sure all the dimensions are consistent
		push!(ax_checks, :( check_axes($ax, $adims, $arr) ||
		error("dimensions $($adims)" * $" of iterator $(ax.args[1])" * $" do not match axes($(arr.args[1]))") ) )

		# code that calculates the strides for each array
		stride = gensym("strides")
		push!(stride_calcs, :( $stride = calc_strides(Val{length($sizevar)}, size($arr), $adims) ) )

		# code that calculates the linear index from the strides
		push!(index_asgns, :( $(esc(idx)) = calc_index($loopvar, $stride)) )
	end

	# replace references of the form A[] with A[iA] where iA is the calculated linear index
	new_body = esc( postwalk(body) do ex
		for (arr, adims, idx) in iterdefs  #zip(arrays, arraydims, indices)
			@capture(ex, $arr[]) && return :( $arr[$idx] )
		end
		return ex
	end)

	loop = quote
		$(ax_checks...)
		$sizevar = get_size($ax)
		$(stride_calcs...)
			for $loopvar in CartesianIndices($sizevar)
				$(index_asgns...)
				$new_body
			end
		end

	return loop
	# return MacroTools.prewalk(rmlines, loop)
end


get_size(x::Integer) = (x,)
get_size(x::Dims) = x
get_size(x::AbstractArray) = size(x)
get_size(x::Axes) = map(a -> last(a) - first(a) + 1, x)


check_axes(arr::AbstractArray, dims, x) = check_axes(axes(arr), dims, x)
check_axes(sz::Dims, dims, x) = check_axes(map(n->Base.OneTo(n), sz), dims, x)
check_axes(n::Int, dims, x) = check_axes((Base.OneTo(n),), dims, x)

function check_axes(ax::Axes, dims::Dims, x::AbstractArray)
	axes(x) == ax[dims]
end


function calc_strides(::Type{Val{N}}, sz::Dims{M}, idx::Dims{M}) where {N,M}
	base_strides = cumprod(ntuple(i -> i>1 ? sz[i-1] : 1, Val(M)))
	accumtuple(base_strides, idx, 0, Val(N), +)
end

@inline calc_index(ci, strides) = 1 + sum(ntuple(i -> (ci[i]-1)*strides[i], Val(length(ci))))

end
