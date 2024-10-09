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
import org.flag4j.arrays.backend.DenseFieldTensorBase;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.io.parsing.ComplexNumberParser;
import org.flag4j.linalg.operations.common.complex.Complex128Operations;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseElemDiv;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseElemMult;
import org.flag4j.linalg.operations.dense.real_field_ops.RealFieldDenseOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.field_ops.DenseCooFieldTensorOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops.RealFieldDenseCooOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.TensorShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A dense complex tensor backed by an array of {@link Complex128}'s.</p>
 *
 * <p>The {@link #entries} of a tensor are mutable but the {@link #shape} is fixed.</p>
 */
public class CTensor extends DenseFieldTensorBase<CTensor, CooCTensor, Complex128> {


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, Field<Complex128>[] entries) {
        super(shape, entries);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, Complex64[] entries) {
        super(shape, new Complex128[entries.length]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128(entries[i]);
    }


    /**
     * Creates a zero tensor with the specified shape.
     *
     * @param shape Shape of this tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex64 fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number parsable by 
     * {@link ComplexNumberParser#parseNumberToComplex128(String)}.
     */
    public CTensor(Shape shape, String fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. Each value in {@code entries} must be formated as a complex number such as:
     * <ul>
     *     <li>"a"</li>
     *     <li>"a + bi", "a - bi", "a + i", or "a - i"</li>
     *     <li>"bi", "i", or "-i"</li>
     * </ul>
     *
     * where "a" and "b" are integers or decimal numbers and white space does not matter.
     */
    public CTensor(Shape shape, String[] entries) {
        super(shape, new Complex128[entries.length]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128(entries[i]);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param tensor Tensor to create copy of.
     */
    public CTensor(CTensor tensor) {
        super(tensor.shape, (Complex128[]) tensor.entries);
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
    public CTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooCTensor toCoo() {
        List<Field<Complex128>> spEntries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = entries.length;
        Field<Complex128> value;

        for(int i=0; i<size; i++) {
            value = entries[i];

            if(!value.isZero()) {
                spEntries.add(value);
                indices.add(shape.getIndices(i));
            }
        }

        return new CooCTensor(shape, spEntries, indices);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CTensor src2 = (CTensor) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Sums this tensor with a dense real tensor.
     * @param b Dense real tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(Tensor b) {
        return new CTensor(shape, RealFieldDenseOperations.add(entries, shape, b.entries, b.shape));
    }


    /**
     * Sums this tensor with a real sparse tensor.
     * @param b Real sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CooTensor b) {
        return (CTensor) RealFieldDenseCooOperations.add(this, b);
    }


    /**
     * Sums this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CooCTensor b) {
        return (CTensor) DenseCooFieldTensorOperations.add(this, b);
    }


    /**
     * Computes difference of this tensor with a dense real tensor.
     * @param b Dense real tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(Tensor b) {
        return new CTensor(shape, RealFieldDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes difference of this tensor with a real sparse tensor.
     * @param b Real sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CooTensor b) {
        return (CTensor) RealFieldDenseCooOperations.sub(this, b);
    }


    /**
     * Computes difference of this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CooCTensor b) {
        return (CTensor) DenseCooFieldTensorOperations.sub(this, b);
    }


    /**
     * Subtracts a specified value from all entries of this tensor and stores the result in this tensor.
     *
     * @param b Value to subtract from all entries of this tensor.
     */
    public void subEq(Double b) {
        RealFieldDenseOperations.subEq(this.entries, b);
    }


    /**
     * Computes the element-wise product of this tensor and a real sparse tensor.
     * @param b Real sparse tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CooCTensor elemMult(CooTensor b) {
        return RealComplexDenseSparseOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of this tensor and a complex sparse tensor.
     * @param b Complex sparse tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CooCTensor elemMult(CooCTensor b) {
        return (CooCTensor) DenseCooFieldTensorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this tensor and a complex dense tensor.
     * @param b Complex dense tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor and {@code b}.
     */
    public CTensor elemDiv(Tensor b) {
        return new CTensor(
                shape,
                RealFieldDenseElemDiv.dispatch(entries, shape, b.entries, b.shape)
        );
    }


    /**
     * Rounds this tensor to the nearest whole number. If the tensor is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     */
    public CTensor round() {
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
    public CTensor round(int precision) {
        return new CTensor(this.shape, Complex128Operations.round(this.entries, precision));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the tensor is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F64} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this tensor with rounded values.
     */
    public CTensor roundToZero() {
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
    public CTensor roundToZero(double threshold) {
        return new CTensor(this.shape, Complex128Operations.roundToZero(this.entries, threshold));
    }


    /**
     * Computes the element-wise sum between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void addEq(Tensor b) {
        RealFieldDenseOperations.addEq(this.entries, this.shape, b.entries, b.shape);
    }


    /**
     * Adds a scalar value to each entry of this tensor and stores the result in this tensor.
     *
     * @param b Scalar value in sum.
     */
    public void addEq(double b) {
        RealFieldDenseOperations.addEq(this.entries, b);
    }


    /**
     * Converts this complex tensor to a real tensor. The resulting tensor will contain only the real components of each entry in
     * this tensor. All imaginary components are ignored.
     * @return A tensor of this same shape as this tensor containing the real components of each entry in this tensor.
     */
    public Tensor toReal() {
        double[] real = new double[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            real[i] = ((Complex128) entries[i]).re;

        return new Tensor(shape, real);
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be broadcastable with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     */
    public CMatrix toMatrix(Shape matShape) {
        ValidateParameters.ensureBroadcastable(shape, matShape);
        ValidateParameters.ensureRank(matShape, rank);

        return new CMatrix(matShape, Arrays.copyOf(entries, entries.length));
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank is larger than 2, it will
     * be flattened to a single row.
     */
    public CMatrix toMatrix() {
        Complex128[] entries = new Complex128[this.entries.length];
        System.arraycopy(this.entries, 0, entries, 0, entries.length);

        if(this.getRank()==2) return new CMatrix(this.shape, entries);
        else return new CMatrix(1, this.entries.length, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public CVector toVector() {
        Complex128[] entries = new Complex128[this.entries.length];
        System.arraycopy(this.entries, 0, entries, 0, entries.length);

        return new CVector(entries);
    }


    /**
     * Computes the element-wise product between this tensor and a real dense tensor.
     * @param b The second tensor in the element-wise product.
     * @return The element-wise product between this tensor and {@code b}.
     */
    public CTensor elemMult(Tensor b) {
        return new CTensor(
                this.shape,
                RealFieldDenseElemMult.dispatch(this.entries, this.shape, b.entries, b.shape)
        );
    }


    /**
     * Computes the element-wise difference between two tensors and stores the result in this tensor.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @throws TensorShapeException If this tensor and {@code b} do not have the same shape.
     */
    public void subEq(Tensor b) {
        RealFieldDenseOperations.subEq(this.entries, this.shape, b.entries, b.shape);
    }


    /**
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flattened entries of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
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

        // Get last entry now
        value = StringUtils.ValueOfRound((Complex128) entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
