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

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldTensorBase;
import org.flag4j.arrays.dense.CTensor;

import java.util.List;


/**
 * <p>Sparse complex tensor stored in coordinate list (COO) format. The entries of this COO tensor are of type {@link Complex128}</p>
 *
 * <p>The non-zero entries and non-zero indices of a COO tensor are mutable but the {@link #shape} and total number of
 * {@link #entries non-zero entries} is fixed.</p>
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
 */
public class CooCTensor extends CooFieldTensorBase<CooCTensor, CTensor, Complex128> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero entries of this tensor of this tensor. If this tensor is dense, this specifies all entries within the
     * tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooCTensor(Shape shape, Complex128[] entries, int[][] indices) {
        super(shape, entries, indices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
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
    public CooCTensor(Shape shape, List<Complex128> entries, List<int[]> indices) {
        super(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex128.ZERO);
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
    public CooCTensor(Shape shape, Complex64[] entries, int[][] indices) {
        super(shape, new Complex128[entries.length], indices);
        setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128(entries[i]);
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
    public CooCTensor(Shape shape) {
        super(shape, new Complex128[0], new int[shape.getRank()][0]);
        super.setZeroElement(Complex128.ZERO);
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
    public CooCTensor makeLikeTensor(Shape shape, Complex128[] entries) {
        return new CooCTensor(shape, entries, indices.clone());
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
    public CooCTensor makeLikeTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
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
    public CooCTensor makeLikeTensor(Shape shape, List<Complex128> entries, List<int[]> indices) {
        return new CooCTensor(shape, entries.toArray(new Complex128[0]), indices.toArray(new int[0][]));
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
    public CTensor makeDenseTensor(Shape shape, Complex128[] entries) {
        return new CTensor(shape, entries);
    }


    /**
     * Converts this sparse tensor to an equivalent dense tensor.
     *
     * @return A dense tensor equivalent to this sparse tensor.
     */
    @Override
    public CTensor toDense() {
        Complex128[] entries = new Complex128[totalEntries().intValueExact()];
        for(int i=0; i<nnz; i++)
            entries[shape.entriesIndex(indices[i])] = this.entries[i];

        return new CTensor(shape, entries);
    }
}
