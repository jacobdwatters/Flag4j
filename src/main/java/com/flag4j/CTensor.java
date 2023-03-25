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
import com.flag4j.core.TensorBase;
import com.flag4j.operations.dense.complex.ComplexDenseEquals;
import com.flag4j.operations.dense.real.RealDenseEquals;
import com.flag4j.operations.dense.real_complex.RealComplexDenseEquals;
import com.flag4j.operations.dense_sparse.complex.ComplexDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real.RealDenseSparseEquals;
import com.flag4j.operations.dense_sparse.real_complex.RealComplexDenseSparseEquals;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * Complex dense tensor.
 */
public class CTensor extends TensorBase<CNumber[]> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fillZeros(super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public CTensor(Shape shape, CNumber fillValue) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);
        ArrayUtils.fill(super.entries, fillValue);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, int[] entries) {
        super(shape, new CNumber[shape.totalEntries().intValue()]);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }

        ArrayUtils.copy2CNumber(entries, super.entries);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * Note, unlike other constructors, the entries parameter is not copied.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public CTensor(Shape shape, CNumber[] entries) {
        super(shape, entries);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(Tensor A) {
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Creates a complex tensor whose shape and entries are specified by another tensor.
     * @param A Tensor specifying shape and entries.
     */
    public CTensor(CTensor A) {
        super(A.shape.copy(), new CNumber[A.totalEntries().intValue()]);
        ArrayUtils.copy2CNumber(A.entries, super.entries);
    }


    /**
     * Checks if an object is equal to this tensor object. Valid object types are: {@link Tensor}, {@link CTensor},
     * {@link SparseTensor}, and {@link SparseCTensor}. These tensors are equal to this tensor if all entries are
     * numerically equal to the corresponding element of this tensor.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors are numerically equivalent and false otherwise.
     */
    @Override
    public boolean equals(Object object) {
        boolean equal;

        if(object instanceof Tensor) {
            Tensor tensor = (Tensor) object;
            equal = RealComplexDenseEquals.tensorEquals(tensor, this);
        } else if(object instanceof CTensor) {
            CTensor tensor = (CTensor) object;
            equal = ComplexDenseEquals.tensorEquals(entries, shape, tensor.entries, tensor.shape);

        } else if(object instanceof SparseTensor) {
            SparseTensor tensor = (SparseTensor) object;
            equal = RealComplexDenseSparseEquals.tensorEquals(this, tensor);

        } else if(object instanceof SparseCTensor) {
            SparseCTensor tensor = (SparseCTensor) object;
            equal = ComplexDenseSparseEquals.tensorEquals(this, tensor);

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Creates a hashcode for this matrix. Note, method adds {@link Arrays#hashCode(double[])} applied on the
     * underlying data array and the underlying shape array.
     * @return The hashcode for this matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(entries)+Arrays.hashCode(shape.dims);
    }

}
