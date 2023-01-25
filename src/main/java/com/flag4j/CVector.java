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

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.core.*;
import com.flag4j.operations.MatrixTranspose;
import com.flag4j.operations.common.complex.AggregateComplex;
import com.flag4j.operations.common.complex.ComplexOperations;
import com.flag4j.operations.common.real.Aggregate;
import com.flag4j.operations.dense.complex.ComplexDenseOperations;
import com.flag4j.operations.dense.complex.ComplexDenseProperties;
import com.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ParameterChecks;

/**
 * Complex dense vector. This class is mostly equivalent to a rank 1 complex tensor.
 */
public class CVector extends VectorBase<CNumber[]> implements
    VectorComparisonsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber>,
    VectorManipulationsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber,
            CMatrix, CMatrix, SparseCMatrix, CMatrix>,
    VectorOperationsMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber,
            CMatrix, CMatrix, SparseCMatrix, CMatrix>,
    VectorPropertiesMixin<CVector, CVector, SparseCVector, CVector, Vector, CNumber> {


    /**
     * Creates a column vector of specified size filled with zeros.
     * @param size Size of the vector.
     */
    public CVector(int size) {
        super(size, new CNumber[size]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, double fillValue) {
        super(size, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector of specified size filled with a specified value.
     * @param size Size of the vector.
     * @param fillValue Value to fill vector with.
     */
    public CVector(int size, CNumber fillValue) {
        super(size, new CNumber[size]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(double[] entries) {
        super(entries.length, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(int[] entries) {
        super(entries.length, new CNumber[entries.length]);
        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Creates a column vector with specified entries.
     * @param entries Entries for this column vector.
     */
    public CVector(CNumber[] entries) {
        super(entries.length, entries);
    }


    /**
     * Constructs a complex vector whose entries and shape are specified by another complex vector.
     * @param a Complex vector to copy.
     */
    public CVector(CVector a) {
        super(a.size(), new CNumber[a.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(a.entries, super.entries);
    }


    /**
     * Checks if this tensor only contains zeros.
     *
     * @return True if this tensor only contains zeros. Otherwise, returns false.
     */
    @Override
    public boolean isZeros() {
        return ArrayUtils.isZeros(this.entries);
    }


    /**
     * Checks if this tensor only contains ones.
     *
     * @return True if this tensor only contains ones. Otherwise, returns false.
     */
    @Override
    public boolean isOnes() {
        return ComplexDenseProperties.isOnes(this.entries);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    @Override
    public void set(double value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        this.entries[indices[0]] = new CNumber(value);
    }


    /**
     * Sets an index of this tensor to a specified value.
     *
     * @param value   Value to set.
     * @param indices The indices of this tensor for which to set the value.
     */
    public void set(CNumber value, int... indices) {
        ParameterChecks.assertArrayLengthsEq(1, indices.length);
        this.entries[indices[0]] = value;
    }


    /**
     * Computes the element-wise addition between two tensors of the same rank.
     *
     * @param B Second tensor in the addition.
     * @return The result of adding the tensor B to this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CVector add(CVector B) {
        return new CVector(ComplexDenseOperations.add(this.entries, this.shape,
                B.entries, B.shape));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(double a) {
        return new CVector(ComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector add(CNumber a) {
        return new CVector(ComplexDenseOperations.add(this.entries, a));
    }


    /**
     * Computes the element-wise subtraction between two tensors of the same rank.
     *
     * @param B Second tensor in element-wise subtraction.
     * @return The result of subtracting the tensor B from this tensor element-wise.
     * @throws IllegalArgumentException If this tensor and B have different shapes.
     */
    @Override
    public CVector sub(CVector B) {
        return new CVector(ComplexDenseOperations.sub(this.entries, this.shape,
                B.entries, B.shape));
    }


    /**
     * Adds specified value to all entries of this tensor.
     *
     * @param a Value to add to all entries of this tensor.
     * @return The result of adding the specified value to each entry of this tensor.
     */
    @Override
    public CVector sub(double a) {
        return new CVector(ComplexDenseOperations.sub(this.entries, a));
    }


    /**
     * Subtracts a specified value from all entries of this tensor.
     *
     * @param a Value to subtract from all entries of this tensor.
     * @return The result of subtracting the specified value from each entry of this tensor.
     */
    @Override
    public CVector sub(CNumber a) {
        return new CVector(ComplexDenseOperations.sub(this.entries, a));
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CVector scalMult(double factor) {
        return new CVector(ComplexDenseOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes scalar multiplication of a tensor.
     *
     * @param factor Scalar value to multiply with tensor.
     * @return The result of multiplying this tensor by the specified scalar.
     */
    @Override
    public CVector scalMult(CNumber factor) {
        return new CVector(ComplexDenseOperations.scalMult(this.entries, factor));
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CVector scalDiv(double divisor) {
        return new CVector(ComplexDenseOperations.scalDiv(this.entries, divisor));
    }


    /**
     * Computes the scalar division of a tensor.
     *
     * @param divisor The scalar value to divide tensor by.
     * @return The result of dividing this tensor by the specified scalar.
     * @throws ArithmeticException If divisor is zero.
     */
    @Override
    public CVector scalDiv(CNumber divisor) {
        return new CVector(ComplexDenseOperations.scalDiv(this.entries, divisor));
    }


    /**
     * Sums together all entries in the tensor.
     *
     * @return The sum of all entries in this tensor.
     */
    @Override
    public CNumber sum() {
        return AggregateComplex.sum(this.entries);
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root i.e. the square root with positive real part.
     */
    @Override
    public CVector sqrt() {
        return new CVector(ComplexOperations.sqrt(this.entries));
    }


    /**
     * Computes the element-wise absolute value/magnitude of a tensor. If the tensor contains complex values, the magnitude will
     * be computed.
     *
     * @return The result of applying an element-wise absolute value/magnitude to this tensor.
     */
    @Override
    public CVector abs() {
        return new CVector(ComplexOperations.abs(this.entries));
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #T()}.
     * This has no effect on a vector.
     * @return The transpose of this tensor.
     */
    @Override
    public CVector transpose() {
        return T();
    }


    /**
     * Computes the transpose of a tensor. Same as {@link #transpose()}.
     * This has no effect on a vector.
     * @return The transpose of this tensor.
     */
    @Override
    public CVector T() {
        return new CVector(this);
    }


    /**
     * Computes the reciprocals, element-wise, of a tensor.
     *
     * @return A tensor containing the reciprocal elements of this tensor.
     * @throws ArithmeticException If this tensor contains any zeros.
     */
    @Override
    public CVector recep() {
        return new CVector(ComplexDenseOperations.recep(this.entries));
    }


    /**
     * Gets the element in this tensor at the specified indices.
     *
     * @param indices Indices of element.
     * @return The element at the specified indices.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this tensor.
     */
    @Override
    public CNumber get(int... indices) {
        return null;
    }


    /**
     * Finds the minimum value in this tensor. If this tensor is complex, then this method finds the smallest value in magnitude.
     *
     * @return The minimum value (smallest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public CNumber min() {
        return null;
    }


    /**
     * Finds the maximum value in this tensor. If this tensor is complex, then this method finds the largest value in magnitude.
     *
     * @return The maximum value (largest in magnitude for a complex valued tensor) in this tensor.
     */
    @Override
    public CNumber max() {
        return null;
    }


    /**
     * Finds the minimum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #min()}.
     *
     * @return The minimum value, in absolute value, in this tensor.
     */
    @Override
    public CNumber minAbs() {
        return null;
    }


    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link #max()}.
     *
     * @return The maximum value, in absolute value, in this tensor.
     */
    @Override
    public CNumber maxAbs() {
        return null;
    }


    /**
     * Finds the indices of the minimum value in this tensor.
     *
     * @return The indices of the minimum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMin() {
        return new int[0];
    }


    /**
     * Finds the indices of the maximum value in this tensor.
     *
     * @return The indices of the maximum value in this tensor. If this value occurs multiple times, the indices of the first
     * entry (in row-major ordering) are returned.
     */
    @Override
    public int[] argMax() {
        return new int[0];
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(double) norm(2)}.
     *
     * @return the 2-norm of this tensor.
     */
    @Override
    public double norm() {
        return 0;
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    @Override
    public double norm(double p) {
        return 0;
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @return The maximum/infinite norm of this tensor.
     */
    @Override
    public double infNorm() {
        return 0;
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
     * Extends a vector a specified number of times to a matrix.
     *
     * @param n    The number of times to extend this vector.
     * @param axis
     * @return A matrix which is the result of extending a vector {@code n} times.
     */
    @Override
    public CMatrix extend(int n, int axis) {
        return null;
    }


    /**
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(Matrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
     */
    @Override
    public CMatrix stack(Vector b) {
        return null;
    }


    /**
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(SparseMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
     */
    @Override
    public CMatrix stack(SparseVector b) {
        return null;
    }


    /**
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(CMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
     */
    @Override
    public CMatrix stack(CVector b) {
        return null;
    }


    /**
     * Stacks two vectors along columns. Note, unlike the {@link MatrixOperationsMixin#stack(SparseCMatrix) stack} method for
     * matrices, the orientation of the vectors <b>IS</b> taken into account (see return section for details).
     *
     * @param b Vector to stack to the bottom of this vector.
     * @return The result of stacking this vector and vector b.<br>
     * - If both vectors are column vectors, then a matrix with 2 columns will be returned.<br>
     * - If both vectors are row vectors, then a matrix with 2 rows will be returned.
     * @throws IllegalArgumentException <br>
     *                                  - If the number of entries in this vector is different from the number of entries in
     *                                  the vector b.<br>
     *                                  - If the vectors are not both row vectors or both column vectors.
     */
    @Override
    public CMatrix stack(SparseCVector b) {
        return null;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber innerProduct(Vector b) {
        return null;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber innerProduct(SparseVector b) {
        return null;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber innerProduct(CVector b) {
        return null;
    }


    /**
     * Computes the inner product between two vectors.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector b.
     * @throws IllegalArgumentException If this vector and vector b do not have the same number of entries.
     */
    @Override
    public CNumber innerProduct(SparseCVector b) {
        return null;
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    @Override
    public CVector cross(Vector b) {
        return null;
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     * @return The result of the vector cross product between this vector and b.
     * @throws IllegalArgumentException If either this vector or b do not have 3 entries.
     */
    @Override
    public CVector cross(CVector b) {
        return null;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrix outerProduct(Vector b) {
        return null;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrix outerProduct(SparseVector b) {
        return null;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrix outerProduct(CVector b) {
        return null;
    }


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and b.
     * @throws IllegalArgumentException If the two vectors do not have the same number of entries.
     */
    @Override
    public CMatrix outerProduct(SparseCVector b) {
        return null;
    }


    /**
     * Converts a vector to an equivalent matrix.
     *
     * @return A matrix equivalent to this vector. This method will respect the orientation of the vector. That is, if
     * this vector is a row vector, then the resulting matrix will have a single row. If this vector is a column vector, then the
     * resulting matrix will have a single column.
     */
    @Override
    public CMatrix toMatrix() {
        return null;
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag for choosing whether to convert this vector to a matrix representing a row or column vector.
     *                    <p>If true, the vector will be converted to a matrix representing a column vector.</p>
     *                    <p>If false, The vector will be converted to a matrix representing a row vector.</p>
     * @return A matrix equivalent to this vector.
     */
    @Override
    public Matrix toMatrix(boolean columVector) {
        return null;
    }

    @Override
    public Tensor toTensor() {
        return null;
    }


    /**
     * Gets the length of a vector.
     *
     * @return The length, i.e. the number of entries, in this vector.
     */
    @Override
    public int length() {
        return 0;
    }
}
