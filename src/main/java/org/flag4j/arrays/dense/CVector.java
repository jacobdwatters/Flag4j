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
import org.flag4j.arrays.backend.field.AbstractDenseFieldVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.common.complex.Complex128Ops;
import org.flag4j.linalg.operations.common.complex.Complex128Properties;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldVectorOperations;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.operations.dense_sparse.coo.field_ops.DenseCooFieldVectorOps;
import org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops.RealFieldDenseCooVectorOps;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.Arrays;

// TODO: Javadoc.
public class CVector extends AbstractDenseFieldVector<CVector, CMatrix, Complex128> {


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Field<Complex128>... entries) {
        super(new Shape(entries.length), entries);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex64... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code data}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(double... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code data}.
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
        Arrays.fill(data, fillValue);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex64 fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(data, new Complex128(fillValue));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, double fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(data, new Complex128(fillValue));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex zero vector with the specified {@code size}.
     * @param size The size of the vector.
     */
    public CVector(int size) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(data, Complex128.ZERO);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a copy of the provided vector.
     * @param vector Vector to create a copy of.
     */
    public CVector(CVector vector) {
        super(vector.shape, vector.data.clone());
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Sets the specified index of this vector to the provided real value.
     * @param val Value to set in this vector.
     * @param index Index to set to {@code val} in this vector.
     * @throws IndexOutOfBoundsException If {@code index} is not in bounds of this vector.
     */
    public void set(double val, int index) {
        set(new Complex128(val), index);
    }


    /**
     * Constructs an empty vector with the specified size. The data of the resulting vector will be
     * all be {@code null}.
     * @param size The size of the vector to construct.
     * @return An empty vector (i.e. filled with {@code null} values) with the specified size.
     */
    public static CVector getEmpty(int size) {
        return new CVector(new Complex128[size]);
    }


    /**
     * Constructs a dense vector with the specified {@code data} of the same type as the vector.
     *
     * @param entries Entries of the dense vector to construct.
     */
    @Override
    public CVector makeLikeTensor(Field<Complex128>[] entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a matrix of similar type to this vector with the specified {@code shape} and {@code data}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of similar type to this vector with the specified {@code shape} and {@code data}.
     */
    @Override
    public CMatrix makeLikeMatrix(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(CooCVector b) {
        return (CVector) DenseCooFieldVectorOps.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(Vector b) {
        RealFieldDenseOps.add(shape, data, b.shape, b.data, data);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector add(CooVector b) {
        return (CVector) RealFieldDenseCooVectorOps.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(CooCVector b) {
        return (CVector) DenseCooFieldVectorOps.sub(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }



    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CVector sub(CooVector b) {
        return (CVector) RealFieldDenseCooVectorOps.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCVector elemMult(CooCVector b) {
        return (CooCVector) DenseCooFieldVectorOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CVector elemMult(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Computes the element-wise multiplication of two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCVector elemMult(CooVector b) {
        Field<Complex128>[] dest = RealFieldDenseCooVectorOps.elemMult(this, b);
        return new CooCVector(shape, dest, b.indices.clone());
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    public CVector div(Vector b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CVector(dest);
    }


    /**
     * Normalizes this vector to a unit length vector.
     *
     * @return This vector normalized to a unit length.
     */
    @Override
    public CVector normalize() {
        return div(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector.
     *
     * @return The magnitude of this vector.
     */
    @Override
    public Complex128 mag() {
        return new Complex128(magAsDouble());
    }


    /**
     * Computes the magnitude of this vector as a double value.
     * @return The magnitude of this vector as a double value.
     */
    public double magAsDouble() {
        double mag = 0;

        for(int i = 0, size = data.length; i < size; i++) {
            Complex128 v = (Complex128) data[i];
            mag += (v.re*v.re + v.im*v.im);
        }

        return Math.sqrt(mag);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCVector makeLikeCooTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        return new CooCVector(shape, entries, indices[0]);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCVector toCoo() {
        return (CooCVector) super.toCoo();
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     *
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     *
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCVector toCoo(double estimatedSparsity) {
        return (CooCVector) super.toCoo(estimatedSparsity);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public CVector makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        ValidateParameters.ensureRank(shape, 1);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        return new CVector(entries);
    }


    /**
     * Converts this complex vector to a real vector. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real vector containing the real components of the data of this vector.
     */
    public Vector toReal() {
        return new Vector(shape, Complex128Ops.toReal(data));
    }


    /**
     * Checks if all data of this vector are real.
     * @return {@code true} if all data of this tensor are real. Otherwise, returns {@code false}.
     */
    public boolean isReal() {
        return Complex128Properties.isReal(data);
    }


    /**
     * Checks if any entry within this vector has non-zero imaginary component.
     * @return {@code true} if any entry of this vector has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Properties.isComplex(data);
    }


    /**
     * Rounds all data within this vector to the specified precision.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A new vector containing the data of this vector rounded to the specified precision.
     */
    public CVector round(int precision) {
        return new CVector(Complex128Ops.round(data, precision));
    }


    /**
     * Sets all elements of this vector to zero if they are within {@code tol} of zero. This is <i>not</i> done in place.
     * @param precision The precision to round to (i.e. the number of decimal places to round to). Must be non-negative.
     * @return A copy of this vector with all data within {@code tol} of zero set to zero.
     */
    public CVector roundToZero(double tolerance) {
        return new CVector(Complex128Ops.roundToZero(data, tolerance));
    }


    /**
     * Compute the inner product of this vector with itself.
     * @return
     */
    public double innerSelf() {
        return DenseFieldVectorOperations.innerSelfProduct(data);
    }


    /**
     * Computes the vector cross product between two vectors.
     *
     * @param b Second vector in the cross product.
     *
     * @return The result of the vector cross product between this vector and {@code b}.
     *
     * @throws IllegalArgumentException If either this vector or {@code b} do not have exactly 3 data.
     */
    public CVector cross(CVector b) {
        if(size != 3 || b.size != 3) {
            throw new LinearAlgebraException("Cross products can only be called vectors of size 3 but got sizes "
                    + size + " and " + b.size);
        }

        Complex128[] entries = new Complex128[3];

        entries[0] = entries[1].mult((Complex128) b.data[2])
                .sub(entries[2].mult((Complex128) b.data[1]));
        entries[1] = entries[2].mult((Complex128) b.data[0])
                .sub(entries[0].mult((Complex128) b.data[2]));
        entries[2] = entries[0].mult((Complex128) b.data[1])
                .sub(entries[1].mult((Complex128) b.data[0]));

        return new CVector(entries);
    }


    /**
     * Computes the element-wise difference between two vectors of the same shape.
     * @param b Second tensor in the element-wise difference.
     * @throws TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public void subEq(Vector b) {
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, data);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return {@code true} if the two vectors have the same shape, are numerically equivalent, and are of type {@link CVector}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CVector src2 = (CVector) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of data to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get data up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(data[i], PrintOptions.getPrecision());
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

        // Get last entry now
        value = StringUtils.ValueOfRound(data[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
