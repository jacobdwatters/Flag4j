/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A dense matrix whose entries are {@link Field} elements.</p>
 *
 * <p>Field matrices have mutable entries but fixed shape.</p>
 *
 * <p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and may have improved performance
 * for some operations.</p>
 *
 * @param <T> Type of the {@link Field field} element for the matrix.
 */
public class FieldMatrix<T extends Field<T>> extends DenseFieldMatrixBase<FieldMatrix<T>, CooFieldMatrix<T>,
        CsrFieldMatrix<T>, FieldVector<T>, T> {


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, Field<T>[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cos, Field<T>[] entries) {
        super(new Shape(rows, cos), entries);
    }



    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, Field<T>[][] entries) {
        super(shape, ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified entries and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Field[shape.totalEntriesIntValueExact()]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a dense field matrix with the specified entries and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, Field<T>[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified entries and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Field[rows*cols]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a matrix of the same type as this matrix with the given the shape filled with the specified fill value.
     *
     * @param shape Shape of the matrix to construct.
     * @param fillValue Value to fill this matrix with.
     *
     * @return A matrix of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, T fillValue) {
        return new FieldMatrix<T>(shape, fillValue);
    }


    /**
     * Constructs a vector of similar type to this matrix with the given {@code entries}.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of similar type to this matrix with the given {@code entries}.
     */
    @Override
    public FieldVector<T> makeLikeVector(Field<T>... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        int rows = numRows;
        int cols = numCols;
        List<Field<T>> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                Field<T> val = entries[rowOffset + j];

                if(!val.isZero()) {
                    sparseEntries.add((T) val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooFieldMatrix<T>(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     * @return A sparse coo matrix equivalent to this matrix.
     * @see #toCoo()
     */
    public CsrFieldMatrix<T> toCsr() {
        // For simplicity convert to a COO matrix as an intermediate.
        return toCoo().toCsr();
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape, Field)
     * @see #I(int, int, Field)
     */
    public static FieldMatrix I(int size, Field fieldValue) {
        return I(size, size, fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static FieldMatrix I(int numRows, int numCols, Field fieldValue) {
        return I(new Shape(numRows, numCols), fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape The shape of the identity-like matrix to construct.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static FieldMatrix I(Shape shape, Field fieldValue) {
        Field[] identityValues = new Field[shape.totalEntriesIntValueExact()];
        Arrays.fill(identityValues, (Field) fieldValue.getZero());
        Field one = (Field) fieldValue.getOne();

        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int i=0, stop=Math.min(rows, cols); i<stop; i++)
            identityValues[i*cols + i] = one;

        return new FieldMatrix(shape, identityValues);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.</p>
     *
     * <p>For large {@code n} values, this method <i>may</i> significantly more efficient than calling
     * {@code #mult(Matrix) this.mult(this)} {@code n} times.</p>
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public FieldMatrix pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return I(numRows, entries[0]);
        if (n == 1) return this;
        if (n == 2) return this.mult(this);

        FieldMatrix<T> result = I(numRows, entries[0]);  // Start with identity matrix
        FieldMatrix<T> base = this;

        // Compute the matrix power efficiently using an "exponentiation by squaring" approach.
        while(n > 0) {
            if((n & 1) == 1)  // If n is odd.
                result = result.mult(base);

            base = base.mult(base);  // Square the base.
            n >>= 1;  // Divide n by 2 (bitwise right shift).
        }

        return result;
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link FieldMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldMatrix<T> src2 = (FieldMatrix<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }
}
