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

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.operations.common.complex.Complex128Operations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.field_ops.DenseCooFieldVectorOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseVectorOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex dense vector whose entries are {@link Complex128}'s.</p>
 *
 * <p>A vector is essentially equivalent to a rank 1 tensor but has some extended functionality and may have improved performance
 * for some operations.</p>
 *
 * <p>CVector's have mutable entries but a fixed size.</p>
 */
public class CVector extends DenseFieldVectorBase<CVector, CMatrix, CooCVector, Complex128> {

    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Field<Complex128>... entries) {
        super(new Shape(entries.length), entries);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex64... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(double... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(int... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex128 fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, fillValue);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex64 fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, new Complex128(fillValue));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, double fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, new Complex128(fillValue));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex zero vector with the specified {@code size}.
     * @param size The size of the vector.
     */
    public CVector(int size) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, Complex128.ZERO);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a copy of the provided vector.
     * @param vector Vector to create a copy of.
     */
    public CVector(CVector vector) {
        super(vector.shape, vector.entries.clone());
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs an empty complex vector with the specified {@code size}. The entries of the resulting vector will be {@code null}.
     * @param size The size of the vector.
     * @return An empty complex vector with the specified {@code size}.
     */
    public static CVector getEmpty(int size) {
        return new CVector(new Complex128[size]);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    @Override
    public CVector makeLikeTensor(int size, Complex128 fillValue) {
        return new CVector(size, fillValue);
    }


    /**
     * Creates a vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    @Override
    public CVector makeLikeTensor(Field<Complex128>... entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a matrix of similar type to this vector with the specified {@code shape} and {@code entries}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of similar type to this vector with the specified {@code shape} and {@code entries}.
     */
    @Override
    public CMatrix makeLikeMatrix(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a sparse vector of similar type to this dense vector.
     *
     * @param size The size of the sparse vector.
     * @param entries The non-zero entries of the sparse vector.
     * @param indices The non-zero indices of the sparse vector.
     *
     * @return A sparse vector of similar type to this dense vector with the specified size, entries, and indices.
     */
    @Override
    public CooCVector makeSparseVector(int size, List<Field<Complex128>> entries, List<Integer> indices) {
        return new CooCVector(size, entries, indices);
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
    public CVector makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        ValidateParameters.ensureRank(shape, 1);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        return new CVector(entries);
    }


    /**
     * Computes the inner product between this vector and itself.
     *
     * @return The inner product between this vector and itself.
     */
    public double innerSelf() {
        double inner = 0;
        for(Field<Complex128> value : entries) {
            Complex128 vCmp = (Complex128) value;
            inner += (vCmp.re*vCmp.re + vCmp.im*vCmp.im);
        }


        return inner;
    }


    /**
     * Converts this complex vector to a real vector. This is done by ignoring the imaginary component of all entries.
     * @return A real vector containing the real components of this complex vectors entries.
     */
    public Vector toReal() {
        double[] real = new double[entries.length];
        for(int i=0, size=entries.length; i<size; i++)
            real[i] = ((Complex128) entries[i]).re;

        return new Vector(shape, real);
    }


    /**
     * Adds a complex dense vector to this vector.
     * @param b Complex dense vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(Vector b) {
        return new CVector(RealComplexDenseOperations.add(entries, shape, b.entries, b.shape));
    }


    /**
     * Adds a real sparse vector to this vector.
     * @param b The real sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CooVector b) {
        return RealComplexDenseSparseVectorOperations.add(this, b);
    }


    /**
     * Adds a complex sparse vector to this vector.
     * @param b The complex sparse vector in the sum.
     * @return The sum of this vector and {@code b}.
     */
    public CVector add(CooCVector b) {
        return (CVector) DenseCooFieldVectorOperations.add(this, b);
    }


    /**
     * Adds a complex-valued scalar to each entry of this vector.
     * @param b The scalar value in the sum.
     * @return The sum of this vector's entries with the scalar value {@code b}.
     */
    public CVector add(Complex128 b) {
        Complex128[] sum = new Complex128[size];

        for(int i=0; i<size; i++)
            sum[i] = b.add((Complex128) entries[i]);

        return new CVector(sum);
    }


    /**
     * Subtracts a complex dense vector from this vector.
     * @param b Complex dense vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(Vector b) {
        return new CVector(RealComplexDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Subtracts a real sparse vector from this vector.
     * @param b The real sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CooVector b) {
        return RealComplexDenseSparseVectorOperations.sub(this, b);
    }


    /**
     * Subtracts a complex sparse vector from this vector.
     * @param b The complex sparse vector in the difference.
     * @return The difference of this vector and {@code b}.
     */
    public CVector sub(CooCVector b) {
        return (CVector) DenseCooFieldVectorOperations.sub(this, b);
    }


    /**
     * Subtracts a scalar from each entry of this tensor. The result is stored in this tensor.
     * @param b The scalar to subtract from each entry of this tensor.
     */
    public void subEq(double b) {
        subEq(new Complex128(b));
    }


    /**
     * Computes the element-wise product of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CVector elemMult(Vector b) {
        return new CVector(RealComplexDenseElemMult.dispatch(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes the element-wise product of this vector and a real sparse vector.
     * @param b The real sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooCVector elemMult(CooVector b) {
        return RealComplexDenseSparseVectorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of this vector and a complex sparse vector.
     * @param b The complex sparse vector in the element-wise product.
     * @return The element-wise product of this vector and {@code b}.
     */
    public CooCVector elemMult(CooCVector b) {
        return (CooCVector) DenseCooFieldVectorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this vector and a complex dense vector.
     * @param b The complex dense vector in the element-wise quotient.
     * @return The element-wise quotient of this vector and {@code b}.
     */
    public CVector div(Vector b) {
        return new CVector(RealComplexDenseElemDiv.dispatch(entries, shape, b.entries, b.shape));
    }


    /**
     * Rounds this tensor to the nearest whole number. If the tensor is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     */
    public CVector round() {
        return round(0);
    }


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    public CVector round(int precision) {
        return new CVector(Complex128Operations.round(entries, precision));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the tensor is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F64} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this tensor with rounded values.
     */
    public CVector roundToZero() {
        return roundToZero(Flag4jConstants.EPS_F64);
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    public CVector roundToZero(double threshold) {
        return new CVector(Complex128Operations.roundToZero(entries, threshold));
    }


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(Vector b) {
        RealComplexDenseOperations.subEq(entries, shape, b.entries, b.shape);
    }


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(Vector b) {
        RealComplexDenseOperations.addEq(entries, shape, b.entries, b.shape);
    }


    /**
     * Sets a value of this vector to {@code val}.
     * @param val Value to set.
     * @param index Index of this vector to set to {@code val}.
     * @throws ArrayIndexOutOfBoundsException If {@code index} is not within this vector.
     */
    public CVector set(double val, int index) {
        return set(new Complex128(val), index);
    }



    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in sum.
     */
    public void addEq(double b) {
        RealComplexDenseOperations.addEq(entries, b);
    }


    /**
     * Generates a human-readable string representation of this vector.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");

        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound((Complex128) entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry.
        value = StringUtils.ValueOfRound((Complex128) entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
