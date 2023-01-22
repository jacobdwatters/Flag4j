/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j;

import com.flag4j.core.*;

import java.util.Arrays;


/**
 * Real dense vector. This class is mostly Equivalent
 */
public class Vector extends VectorBase<double[]> implements 
        VectorComparisonsMixin<Vector, Vector, SparseVector, CVector, Vector, Double>,
        VectorManipulationsMixin<Vector, Vector, SparseVector, CVector, Vector, Double,
            Matrix, Matrix, SparseMatrix, CMatrix> {


//    VectorComparisonsMixin<Vector, Vector, SparseVector, CVector, Vector, Double>,
//    VectorManipulationsMixin<Vector, Vector, SparseVector, CVector, Vector, Double,
//            Matrix, Matrix, SparseMatrix, CMatrix>,
//    VectorOperationsMixin<Vector, Vector, SparseVector, CVector, Vector, Double,
//            Matrix, Matrix, SparseMatrix, CMatrix>,
//    VectorPropertiesMixin<Vector, Vector, SparseVector, CVector, Vector, Double>

    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public Vector(int size) {
        super(size, new double[size]);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public Vector(int size, double fillValue) {
        super(size, new double[size]);
        Arrays.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(double[] entries) {
        super(entries.length, entries.clone());
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public Vector(int[] entries) {
        super(entries.length, new double[entries.length]);

        for(int i=0; i<entries.length; i++) {
            super.entries[i] = entries[i];
        }
    }


    /**
     * Creates a vector from another vector. This essentially copies the vector.
     * @param a Vector to make copy of.
     */
    public Vector(Vector a) {
        super(a.entries.length, a.entries.clone());
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return false;
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return false;
    }


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     *
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(Vector b) {
        return false;
    }


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     *
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(SparseVector b) {
        return false;
    }


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     *
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(CVector b) {
        return false;
    }


    /**
     * Checks if two vectors are equal (element-wise). This method <b>DOES</b> take into consideration the
     * orientation of the vectors.
     *
     * @param b Second vector in the equality.
     * @return True if this vector and vector b are equivalent element-wise. Otherwise, returns false.
     */
    @Override
    public boolean equals(SparseCVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    @Override
    public boolean sameShape(Vector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    @Override
    public boolean sameShape(SparseVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    @Override
    public boolean sameShape(CVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same shape. This method <b>DOES</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same orientation and length. Otherwise, returns false.
     */
    @Override
    public boolean sameShape(SparseCVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    @Override
    public boolean sameSize(Vector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    @Override
    public boolean sameSize(SparseVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    @Override
    public boolean sameSize(CVector b) {
        return false;
    }


    /**
     * Checks if two vectors have the same length. This method <b>DOES NOT</b> take the two vectors orientations into account.
     *
     * @param b Vector to compare to this vector.
     * @return True if this vector and vector b have the same length. Otherwise, returns false.
     */
    @Override
    public boolean sameSize(SparseCVector b) {
        return false;
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public void set(double value, int... indices) {

    }


    /**
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n The number of times to extend this vector.
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public Matrix extend(int n) {
        return null;
    }
}
