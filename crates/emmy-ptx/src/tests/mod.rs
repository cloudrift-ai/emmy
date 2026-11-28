//! The rule tests: kernels from the text form, lowered, assembled by ptxas wherever it is installed, and read one
//! instruction a line. Each kernel is printed for an AMD card as well and assembled by `llc` wherever that is.

mod amdgcn;
mod bf16;
mod common;
mod cooperative;
mod fragment;
mod hopper;
mod math;
mod rules;
mod tensor;
