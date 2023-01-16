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

import com.flag4j.core.TensorBase;
import com.flag4j.operations.concurrency.util.ErrorMessages;

import java.util.Arrays;


/**
 * Real Dense Tensor. Can be any rank.
 */
public class Tensor extends TensorBase<double[]> {


    /**
     * Constructs a tensor with given shape filled with zeros.
     * @param shape Shape of the tensor.
     */
    public Tensor(Shape shape) {
        super(shape, new double[shape.totalEntries().intValue()]);
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param fillValue Value to fill tensor with.
     */
    public Tensor(Shape shape, double fillValue) {
        super(shape, new double[shape.totalEntries().intValue()]);

        for(int i=0; i<super.totalEntries().intValue(); i++) {
            super.entries[i] = fillValue;
        }
    }

    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, double[] entries) {
        super(shape, entries);

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor with given shape filled with specified values.
     * @param shape Shape of the tensor.
     * @param entries Entries of the vector.
     * @throws IllegalArgumentException If the shape does not match the number of entries.
     */
    public Tensor(Shape shape, int[] entries) {
        super(shape, Arrays.stream(entries).asDoubleStream().toArray());

        if(entries.length != super.totalEntries().intValue()) {
            throw new IllegalArgumentException(ErrorMessages.shapeEntriesError(shape, entries.length));
        }
    }


    /**
     * Constructs a tensor from another tensor. This effectively copies the tensor.
     * @param A tensor to copy.
     */
    public Tensor(Tensor A) {
        super(A.shape.copy(), A.entries.clone());
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a matrix.
     * @param A Matrix to copy to tensor.
     */
    public Tensor(Matrix A) {
        super(A.shape.copy(), A.entries.clone());
    }


    /**
     * Constructs a tensor whose shape and entries are specified by a vector.
     * @param A Vector to copy to tensor.
     */
    public Tensor(Vector A) {
        super(A.getOrientedShape(), A.entries.clone());
    }
}
