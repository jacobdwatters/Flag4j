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


import org.flag4j.algebraic_structures.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldTensor;
import org.flag4j.arrays.dense.FieldTensor;
import org.flag4j.arrays.dense.FieldVector;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.sparse.coo.field_ops.CooFieldEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Sparse tensor stored in coordinate list (COO) format. The data of this COO tensor are elements of a {@link Field}
 *
 * <p>The non-zero data and non-zero indices of a COO tensor are mutable but the shape and total number of non-zero data is
 * fixed.
 *
 * <p>Sparse tensors allow for the efficient storage of and ops on tensors that contain many zero values.
 *
 * <p>COO tensors are optimized for hyper-sparse tensors (i.e. tensors which contain almost all zeros relative to the size of the
 * tensor).
 *
 * <p>A sparse COO tensor is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the tensor.</li>
 *     <li>The non-zero {@link #data} of the tensor. All other data in the tensor are
 *     assumed to be zero. Zero value can also explicitly be stored in {@link #data}.</li>
 *     <li><p>The {@link #indices} of the non-zero value in the sparse tensor. Many ops assume indices to be sorted in a
 *     row-major format (i.e. last index increased fastest) but often this is not explicitly verified.
 *
 *     <p>The {@link #indices} array has shape {@code (nnz, rank)} where {@link #nnz} is the number of non-zero data in this
 *     sparse tensor and {@code rank} is the {@link #getRank() tensor rank} of the tensor. This means {@code indices[i]} is the ND
 *     index of {@code data[i]}.
 *     </li>
 * </ul>
 *
 * @param <T> Type of the {@link #data} of this tensor.
 */
public class CooFieldTensor<T extends Field<T>>
        extends AbstractCooFieldTensor<CooFieldTensor<T>, FieldTensor<T>, T> {

    private static final long serialVersionUID = 1L;

    /**
     * creates a tensor with the specified data and shape.
     *
     * @param shape shape of this tensor.
     * @param entries non-zero data of this tensor of this tensor. if this tensor is dense, this specifies all data within the
     * tensor.
     * if this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, T[] entries, int[][] indices) {
        super(shape, entries, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the specified shape and non-zero data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Non-zero data of the tensor to construct.
     * @param indices Indices of the non-zero data of the tensor.
     *
     * @return A tensor of the same type as this tensor with the specified shape and non-zero data.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, T[] entries, int[][] indices) {
        return new CooFieldTensor<>(shape, entries, indices);
    }


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this tensor of this tensor. If this tensor is dense, this specifies all data within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     * @param indices
     */
    public CooFieldTensor(Shape shape, List<T> entries, List<int[]> indices) {
        super(shape, (T[]) entries.toArray(), indices.toArray(new int[0][]));
    }


    /**
     * Sets the element of this tensor at the specified indices.
     *
     * @param value New value to set the specified index of this tensor to.
     * @param index Indices of the element to set.
     *
     * @return A copy of this tensor with the updated value is returned.
     *
     * @throws IndexOutOfBoundsException If {@code indices} is not within the bounds of this tensor.
     */
    @Override
    public CooFieldTensor<T> set(T value, int... index) {
        ValidateParameters.validateTensorIndex(shape, index);
        CooFieldTensor<T> dest;

        // Check if value already exists in tensor.
        int idx = -1;
        for(int i=0; i<indices.length; i++) {
            if(Arrays.equals(indices[i], index)) {
                idx = i;
                break; // Found in tensor, no need to continue.
            }
        }

        if(idx > -1) {
            // Copy data and set new value.
            dest = new CooFieldTensor<T>(shape, data.clone(), ArrayUtils.deepCopy(indices, null));
            dest.data[idx] = value;
            dest.indices[idx] = index;
        } else {
            // Copy old indices and insert new one.
            int[][] newIndices = new int[indices.length + 1][getRank()];
            ArrayUtils.deepCopy(indices, newIndices);
            newIndices[indices.length] = index;

            // Copy old data and insert new one.
            T[] newEntries = Arrays.copyOf(data, data.length+1);
            newEntries[newEntries.length-1] = value;

            dest = new CooFieldTensor<T>(shape, newEntries, newIndices);
            dest.sortIndices();
        }

        return dest;
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and data.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Entries of the spares tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the same indices as this sparse tensor and with the provided
     * the shape and data.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, T[] entries) {
        return new CooFieldTensor(shape, entries, ArrayUtils.deepCopy(indices, null));
    }


    /**
     * Constructs a sparse tensor of the same type as this tensor with the given the shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the sparse tensor to construct.
     * @param entries Non-zero data of the sparse tensor to construct.
     * @param indices Non-zero indices of the sparse tensor to construct.
     *
     * @return A sparse tensor of the same type as this tensor with the given the shape and data.
     */
    @Override
    public CooFieldTensor<T> makeLikeTensor(Shape shape, List<T> entries, List<int[]> indices) {
        return new CooFieldTensor(shape, entries, indices);
    }


    /**
     * Constructs a dense tensor that is a similar type as this sparse COO tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries The data of the dense tensor to construct.
     *
     * @return A dense tensor that is a similar type as this sparse COO tensor.
     */
    @Override
    public FieldTensor<T> makeLikeDenseTensor(Shape shape, T[] entries) {
        return new FieldTensor<T>(shape, entries);
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
        T[] entries = (T[]) new Field[totalEntries().intValueExact()];

        for(int i = 0; i< nnz; i++)
            entries[shape.getFlatIndex(indices[i])] = this.data[i];

        return new FieldTensor<T>(shape, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public FieldVector<T> toVector() {
        return new FieldVector<T>(this.data.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be {@link ValidateParameters#ensureTotalEntriesEqual(Shape, Shape) broadcastable}
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public CooFieldMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureRank(matShape, 2);

        CooFieldTensor<T> t = reshape(matShape); // Reshape as rank 2 tensor. Broadcastable check is made here.
        int[][] tIndices = RealDenseTranspose.standardIntMatrix(t.indices);

        return new CooFieldMatrix<T>(matShape, t.data.clone(), tIndices[0], tIndices[1]);
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public CooFieldMatrix<T> toMatrix() {
        CooFieldMatrix<T> mat;

        if(this.getRank()==2) {
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(this.indices);
            mat = new CooFieldMatrix<T>(shape, data.clone(), tIndices[0], tIndices[1]);
        } else {
            CooFieldTensor<T> flat = reshape(new Shape(1, shape.totalEntriesIntValueExact()));
            int[][] tIndices = RealDenseTranspose.standardIntMatrix(flat.indices);
            mat = new CooFieldMatrix<T>(flat.shape, flat.data.clone(), tIndices[0], tIndices[0]);
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

        return CooFieldEquals.cooTensorEquals(this, src2);
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for(int i=0; i<nnz; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Arrays.hashCode(indices[i]);
            }
        }

        return result;
    }


    /**
     * <p>Formats this sparse COO tensor as a human-readable string specifying the full shape,
     * non-zero data, and non-zero indices.
     *
     * @return A human-readable string specifying the full shape, non-zero data, and non-zero indices of this tensor.
     */
    public String toString() {
        int maxCols = PrintOptions.getMaxColumns();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();
        boolean centering = PrintOptions.useCentering();

        StringBuilder sb = new StringBuilder();

        sb.append("Shape: " + shape + "\n");
        sb.append("nnz: ").append(nnz).append("\n");
        sb.append("Non-zero Entries: " + PrettyPrint.abbreviatedArray(data, maxCols, padding, precision, centering) + "\n");
        sb.append("Non-zero Indices: " +
                PrettyPrint.abbreviatedArray(indices, PrintOptions.getMaxRows(), maxCols, padding, 20, centering));

        return sb.toString();
    }
}
