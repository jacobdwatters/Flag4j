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
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldTensorBase;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.parsing.ComplexNumberParser;
import org.flag4j.operations.dense.complex.ComplexDenseElemMult;
import org.flag4j.operations.dense.complex.ComplexDenseOperations;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.complex.ComplexDenseSparseOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;

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
    public CTensor(Shape shape, Complex128[] entries) {
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
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
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
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
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
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CTensor makeLikeTensor(Shape shape, Complex128[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooCTensor toCoo() {
        List<Complex128> spEntries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = entries.length;
        Complex128 value;

        for(int i=0; i<size; i++) {
            value = entries[i];

            if(value.isZero()) {
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
        return new CTensor(shape, RealComplexDenseOperations.add(entries, shape, b.entries, b.shape));
    }


    /**
     * Sums this tensor with a real sparse tensor.
     * @param b Real sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CooTensor b) {
        return RealComplexDenseSparseOperations.add(this, b);
    }


    /**
     * Sums this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CooCTensor b) {
        return ComplexDenseSparseOperations.add(this, b);
    }


    /**
     * Adds a complex-valued scalar to each entry of this tensor.
     * @param b Scalar to add to each entry of this tensor.
     * @return Tensor containing sum of all entries of this tensor with {@code b}.
     */
    public CTensor add(Complex128 b) {
        return new CTensor(shape, ComplexDenseOperations.add(entries, b));
    }


    /**
     * Computes difference of this tensor with a dense real tensor.
     * @param b Dense real tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(Tensor b) {
        return new CTensor(shape, RealComplexDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes difference of this tensor with a real sparse tensor.
     * @param b Real sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CooTensor b) {
        return RealComplexDenseSparseOperations.sub(this, b);
    }


    /**
     * Computes difference of this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CooCTensor b) {
        return ComplexDenseSparseOperations.sub(this, b);
    }


    /**
     * Computes the element-wise product of this tensor and a complex dense tensor.
     * @param b Complex dense tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CTensor elemMult(CTensor b) {
        return new CTensor(
                shape,
                ComplexDenseElemMult.dispatch(b.entries, b.shape, entries, shape)
        );
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
        return ComplexDenseSparseOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this tensor and a complex dense tensor.
     * @param b Complex dense tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor and {@code b}.
     */
    public CTensor elemDiv(Tensor b) {
        return new CTensor(
                shape,
                RealComplexDenseElemDiv.dispatch(entries, shape, b.entries, b.shape)
        );
    }
}
