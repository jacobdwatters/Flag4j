package com.flag4j.core;

import com.flag4j.Shape;


/**
 * The base class for all real matrices.
 */
public abstract class RealMatrixBase extends MatrixBase<double[]> {


    /**
     * Constructs a basic matrix with a given shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     * @throws IllegalArgumentException If the shape parameter is not of rank 2.
     */
    public RealMatrixBase(Shape shape, double[] entries) {
        super(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent complex matrix.
     * @return A complex matrix with equivalent real part and zero imaginary part.
     */
    public abstract ComplexMatrixBase toComplex();
}
