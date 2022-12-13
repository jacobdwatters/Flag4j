package com.flag4j.core;


import com.flag4j.Shape;
import com.flag4j.complex_numbers.CNumber;

public abstract class ComplexMatrixBase extends MatrixBase<CNumber[]> {


    /**
     * Constructs a basic matrix with a given shape.
     *
     * @param shape   Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public ComplexMatrixBase(Shape shape, CNumber[] entries) {
        super(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent real matrix. Imaginary components are ignored.
     * @return A real matrix with equivalent real parts.
     */
    public abstract RealMatrixBase toReal();
}
