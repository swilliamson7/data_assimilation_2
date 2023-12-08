using Enzyme
Enzyme.API.printall!(true)
include("ThreeMassSpring.jl")

#doesn't fail 
ThreeMassSpring.enzyme_check_param()

#fail 
ThreeMassSpring.enzyme_check_param()