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

package org.flag4j.arrays.sparse;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldTensorBase;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.operations.sparse.coo.field_ops.SparseFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Sparse tensor stored in coordinate list (COO) format. The entries of this COO tensor are elements of a {@link Field}</p>
 *
 * <p>The non-zero entries and non-zero indices of a COO tensor are mutable but the shape and total number of non-zero entries is
 * fixed.</p>
 *
 * <p>Sparse tensors allow for the efficient storage of and operations on tensors that contain many zero values.</p>
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).</p>
 *
 * <p>A sparse COO tensor is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #entries} of the tensor. All other entries in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #entries}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many operations assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.</p>
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero entries in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code entries[i]}.</p>
 *     </li>
 * </ul>
 *
 * @param <T> Type of the {@link #entries} of this tensor.
 */
public class CooFieldTensor<T extends Field<T>>
        extends CooFieldTensorBase<CooFieldTensor<T>, FieldTensor<T>, T> {

    /**
     * creates a tensor with the specified entries and shape.
     *
     * @param shape shape of this tensor.
     * @param entries non-zero entries of this tensor of this tensor. if this tensor is dense, this specifies all entries within the
     * tensor.
     * if this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, T[] entries, int[][] indices) {
        super(shape, entries, indices);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, List<Field<T>> entries, List<int[]> indices) {
        super(shape, (T[]) entries.toArray(new Field[0]), indices.toArray(new int[0][]));
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Entries of the spares tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and entries.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooFieldTensor(shape, entries, ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, T[] entries, int[][] indices) {
        return new CooFieldTensor(shape, entries, indices);
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero entries, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero entries of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, List<T> entries, List<int[]> indices) {
        return new CooFieldTensor(shape, entries, indices);
    }


    /**
     * Makes a dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified shape and entries which is a similar type to this sparse tensor.
     */
    @Override
    public FieldTensor<T> makeDenseTensor(Shape shape, T[] entries) {
        return new FieldTensor(shape, entries);
    }


    /**
     * The sparsity of this sparse tensor. That is, the percentage of elements in this tensor which are zero as a decimal.
     *
     * @return The density of this sparse tensor.
     */
    @Override
    public double sparsity() {
        if(nnz == 0) return 1.0;

        BigInteger totalEntries = totalEntries();
        BigDecimal sparsity = new BigDecimal(totalEntries).subtract(BigDecimal.valueOf(nnz));
        sparsity = sparsity.divide(new BigDecimal(totalEntries), 50, RoundingMode.HALF_UP);

        return sparsity.doubleValue();
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public FieldTensor<T> toDense() {
        Field<T>[] entries = new Field[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[shape.entriesIndex(indices[i])] = this.entries[i];

        return new FieldTensor<T>(shape, (T[]) entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public FieldVector<T> toVector() {
        return new FieldVector<T>(this.entries.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be {@link ParameterChecks#ensureBroadcastable(Shape, Shape) broadcastable}
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public FieldMatrix<T> toMatrix(Shape matShape) {
        ParameterChecks.ensureBroadcastable(shape, matShape);
        ParameterChecks.ensureRank(matShape, 2);

        return new FieldMatrix<T>(matShape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public FieldMatrix<T> toMatrix() {
        FieldMatrix<T> mat;

        if(this.getRank()==2) {
            mat = new FieldMatrix<T>(this.shape, this.entries.clone());
        } else {
            mat = new FieldMatrix<T>(1, this.entries.length, this.entries.clone());
        }

        return mat;
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CooFieldTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CooFieldTensor<T> src2 = (CooFieldTensor<T>) object;

        return SparseFieldEquals.cooTensorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for(int i=0; i<nnz; i++) {
            if (!entries[i].isZero()) {
                result = 31*result + entries[i].hashCode();
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }
}
