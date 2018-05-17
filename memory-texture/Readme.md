## Texture Memory

# Purpose
This very basic example demonstrates how to use the read-only texture memory through pointer assignemnt. We also demonstrate that without explicit use of the texture memory, and only with declaring read-only variables as `intent(in)`, we obtain the same bandwidth. This shows that in CUDA Fortran, read-only variables are transfered through the texture memory if declared correctly. 

# Build and Dependencies
This example compiles with the `PGI 18.4` CUDA Fortran compiler. To compile, just execute the bash script `compile.sh`, and to run execute `main.exe`.

## Tip
A big lesson learned from the definition of the texture pointer is not to assign it to a `null()` pointer at the declaration statement. As an example, the following shows the wrong and correct practices:
```fortran
real, pointer, dimension(:), texture :: tex_bad => null()
real, pointer, dimension(:), texture :: tex_OK
```
In fact, ignoring this made me stuck for days, trying to understand why I got the out-of-bound memory errors. So, a good lesson learned.

