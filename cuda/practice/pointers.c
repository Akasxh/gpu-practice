// A bit about size_t : size_t is an unsigned integer type that is used to represent the size of an object.
// It is defined in the stddef.h header file.
// It is used to represent the size of a object by the compile and is biggest size of the data type that can be represented by the compiler.

#include <stdio.h>

int main() {
    size_t size = sizeof(int);
    printf("Size of int: %zu\n", size);
    return 0;
}

// Output:
// Size of int: 4

// ------------------------------------------------------------

// int main() {
//     int x[] = {1, 2, 3, 4, 5};
//     int *ptr = x;
//     int *ptr2[5];

//     printf("Value of ptr: %p\n", ptr);
//     printf("Value of *ptr: %d\n", *ptr);
//     printf("Value of x[0]: %d\n", x[0]);

//     for (int i = 0; i < 5; i++) {
//         ptr2[i] = &x[i];
//     }

//     printf("Value of ptr2: %p\n", ptr2);
//     printf("Value of *ptr2: %p\n", *ptr2);
//     printf("Value of x[0]: %d\n", x[0]);

//     return 0;
// }

// Output:
// Value of ptr: 0x7ffd4f4108a0
// Value of *ptr: 1
// Value of x[0]: 1
// Value of ptr2: 0x7ffd4f4108a0
// Value of *ptr2: 0x7ffd4f4108a0
// Value of x[0]: 1