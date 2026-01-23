# How to get the SORT algorithm to work
1. Add the SORT.h and the hungarian.h wherever you want your header files to be.
2. Add the SORT.cpp and the hungarian.cpp to wherever you want your source files to be.
3. Wherever the struct named "Detection" is defined, add another integer member called object_id and set it to -1. (`int object_id = -1`).

All object ids are initialized to -1 and then is assigned a positive integer ID. GL