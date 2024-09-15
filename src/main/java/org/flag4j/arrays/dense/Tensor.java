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


import org.flag4j.arrays.backend.DensePrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.TensorOverSemiRing;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays.Shape;
import org.flag4j.linalg.TensorInvertOld;
import org.flag4j.operations.dense.real.RealDenseEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real dense tensor backed by a primitive double array.</p>
 * <p>The {@link #entries} of a Tensor are mutable but the {@link #shape} is fixed.</p>
 */
public class Tensor extends DensePrimitiveDoubleTensorBase<Tensor, CooTensor> {

    /**
     * Creates a zero tensor with the shape.
     *
     * @param shape Shape of this tensor.
     */
    public Tensor(Shape shape) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
    }


    /**
     * Creates a tensor with the specified shape filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public Tensor(Shape shape, double fillValue) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, double[] entries) {
        super(shape, entries);
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
    public Tensor makeLikeTensor(Shape shape, double[] entries) {
        return new Tensor(shape, entries);
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link Tensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        TensorOld src2 = (TensorOld) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.inv(numIndices)} such that
     * {@link #tensorDot(TensorOverSemiRing, int)} this.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor dot product
     * operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I, numIndices).equals(this)}.</p>
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #inv()
     */
    public Tensor inv(int numIndices) {
        return TensorInvertOld.inv(this, numIndices);
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.inv()} such that
     * {@link #tensorDot(TensorOverSemiRing) this.tensorDot(X)} is the 'identity' tensor for the tensor dot product
     * operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code this.tensorDot(I).equals(this)}.</p>
     *
     * <p>Equivalent to {@link #inv(int) inv(2)}.</p>
     *
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @see #inv(int)
     */
    public Tensor inv() {
        return inv(2);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse tensor equivalent to this dense tensor.
     */
    @Override
    public CooTensor toCoo() {
        List<Double> SparseEntries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = entries.length;
        double value;

        for(int i=0; i<size; i++) {
            value = entries[i];

            if(value != 0) {
                SparseEntries.add(value);
                indices.add(shape.getIndices(i));
            }
        }

        return new CooTensor(shape, ArrayUtils.fromDoubleList(SparseEntries), indices.toArray(new int[0][]));
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public Vector toVector() {
        return new Vector(this.entries.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be {@link ParameterChecks#ensureBroadcastable(Shape, Shape) broadcastable}
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public Matrix toMatrix(Shape matShape) {
        ParameterChecks.ensureBroadcastable(shape, matShape);
        ParameterChecks.ensureRank(matShape, 2);

        return new Matrix(matShape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public Matrix toMatrix() {
        Matrix mat;

        if(this.getRank()==2) {
            mat = new Matrix(this.shape, this.entries.clone());
        } else {
            mat = new Matrix(1, this.entries.length, this.entries.clone());
        }

        return mat;
    }
}
