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
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.CooFieldVectorBase;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ParameterChecks;

import java.util.List;


/**
 * <p>A complex sparse vector stored in coordinate list (COO) format. The {@link #entries} of this COO vector are
 * {@link Complex128}'s.</p>
 *
 * <p>The {@link #entries non-zero entries} and {@link #indices non-zero indices} of a COO vector are mutable but the {@link #shape}
 * and total number of non-zero entries is fixed.</p>
 *
 * <p>Sparse vectors allow for the efficient storage of and operations on vectors that contain many zero values.</p>
 *
 * <p>COO vectors are optimized for hyper-sparse vectors (i.e. vectors which contain almost all zeros relative to the size of the
 * vector).</p>
 *
 * <p>A sparse COO vector is stored as:</p>
 * <ul>
 *     <li>The full {@link #shape}/{@link #size} of the vector.</li>
 *     <li>The non-zero {@link #entries} of the vector. All other entries in the vector are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #entries}.</li>
 *     <li>The {@link #indices} of the non-zero values in the sparse vector.</li>
 * </ul>
 *
 * <p>Note: many operations assume that the entries of the COO vector are sorted lexicographically. However, this is not explicitly
 * verified. Every operation implemented in this class will preserve the lexicographical sorting.</p>
 *
 * <p>If indices need to be sorted for any reason, call {@link #sortIndices()}.</p>
 */
public class CooCVector extends CooFieldVectorBase<CooCVector, CooCMatrix, CVector, CMatrix, Complex128> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param size
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @param indices
     */
    public CooCVector(int size, Complex128[] entries, int[] indices) {
        super(size, entries, indices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a complex COO vector with the specified size, non-zero entries, and non-zero indices.
     * @param size The size of the vector.
     * @param entries The non-zero entries of the vector.
     * @param indices The indices of the non-zero entries.
     */
    public CooCVector(int size, List<Complex128> entries, List<Integer> indices) {
        super(size, entries.toArray(new Complex128[0]), ArrayUtils.fromIntegerList(indices));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero vector of the specified {@code size}.
     * @param size The size of the vector.
     */
    public CooCVector(int size) {
        super(size, new Complex128[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    @Override
    public CooCVector makeLikeTensor(int size, Complex128[] entries, int[] indices) {
        return new CooCVector(size, entries, indices);
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and the same
     * non-zero indices as this vector.
     */
    @Override
    public CooCVector makeLikeTensor(int size, Complex128[] entries) {
        return new CooCVector(size, entries, indices.clone());
    }


    /**
     * Constructs a sparse COO vector of the same type as this tensor with the specified {@code size}, non-zero entries, and non-zero indices.
     *
     * @param size Size of the sparse COO vector.
     * @param entries Non-zero entries of the sparse COO vector.
     * @param indices Non-zero indices of the sparse COO vector.
     *
     * @return A sparse COO vector of the same type as this vector with the specified {@code size}, non-zero entries,
     * and non-zero indices.
     */
    @Override
    public CooCVector makeLikeTensor(int size, List<Complex128> entries, List<Integer> indices) {
        return new CooCVector(size, entries, indices);
    }


    /**
     * Constructs a dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     *
     * @param entries The entries of the dense vector.
     *
     * @return A dense vector which is of a similar type to this sparse COO vector containing the specified {@code entries}.
     */
    @Override
    public CVector makeLikeDenseTensor(Complex128... entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a sparse matrix which is of a similar type to this sparse COO vector with the specified {@code shape}, non-zero
     * entries, non-zero row indices, and non-zero column indices.
     *
     * @param shape Shape of the matrix.
     * @param entries The non-zero indices of the matrix.
     * @param rowIndices The row indices of the non-zero entries.
     * @param colIndices The column indices of the non-zero entries.
     *
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    @Override
    public CooCMatrix makeLikeMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     *
     * @param shape Shape of the dense matrix.
     * @param entries The entries of the dense matrix.
     *
     * @return A dense matrix which is of a similar type to this sparse COO vector with the specified {@code shape} and containing
     * the specified {@code entries}.
     */
    @Override
    public CMatrix makeLikeDenseMatrix(Shape shape, Complex128... entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a sparse COO vector of the specified size filled with zeros.
     *
     * @param size The size of the vector to construct.
     *
     * @return A sparse COO vector of the specified size filled with zeros.
     */
    @Override
    public CooCVector makeZeroVector(int size) {
        return new CooCVector(size);
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
    public CooCVector makeLikeTensor(Shape shape, Complex128[] entries) {
        ParameterChecks.ensureRank(shape, 1);
        return new CooCVector(shape.get(0), entries, indices.clone());
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     *
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     * <p>If {@code true}, the vector will be converted to a matrix representing a column vector.</p>
     * <p>If {@code false}, The vector will be converted to a matrix representing a row vector.</p>
     *
     * @return A matrix equivalent to this vector.
     */
    @Override
    public CooCMatrix toMatrix(boolean columVector) {
        if(columVector) {
            // Convert to column vector
            int[] rowIndices = indices.clone();
            int[] colIndices = new int[entries.length];

            return new CooCMatrix(this.size, 1, entries.clone(), rowIndices, colIndices);
        } else {
            // Convert to row vector.
            int[] rowIndices = new int[entries.length];
            int[] colIndices = indices.clone();

            return new CooCMatrix(1, this.size, entries.clone(), rowIndices, colIndices);
        }
    }
}
