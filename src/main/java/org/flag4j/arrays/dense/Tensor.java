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
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DensePrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.TensorOverSemiRing;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.TensorInvert;
import org.flag4j.operations.common.complex.Complex128Operations;
import org.flag4j.operations.dense.complex.ComplexDenseOperations;
import org.flag4j.operations.dense.real.RealDenseEquals;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemDiv;
import org.flag4j.operations.dense.real_complex.RealComplexDenseElemMult;
import org.flag4j.operations.dense.real_complex.RealComplexDenseOperations;
import org.flag4j.operations.dense_sparse.coo.real.RealDenseSparseTensorOperations;
import org.flag4j.operations.dense_sparse.coo.real_complex.RealComplexDenseSparseOperations;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real dense tensor backed by a primitive double array.</p>
 *
 * <p>A tensor is a multi-dimensional array. If N indices are required to uniquely identify all elements of a tensor, then the
 * tensor is considered an N-dimensional tensor/array or a rank-N tensor.</p>
 *
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
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, int[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.asDouble(entries, this.entries);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param src The tensor to make a copy of.
     */
    public Tensor(Tensor src) {
        super(src.shape, src.entries.clone());
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, Double[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.unbox(entries, super.entries);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, Integer[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.asDouble(entries, super.entries);
    }


    /**
     * Converts a matrix to an equivalent tensor.
     *
     * @param mat Matrix to convert to a tensor.
     */
    public Tensor(Matrix mat) {
        super(mat.shape, mat.entries.clone());
    }


    /**
     * Converts a matrix to an equivalent tensor.
     *
     * @param mat Matrix to convert to a tensor.
     */
    public Tensor(Vector mat) {
        super(mat.shape, mat.entries.clone());
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

        Tensor src2 = (Tensor) object;

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
        return TensorInvert.inv(this, numIndices);
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
     * @param matShape Shape of the resulting matrix. Must be {@link ValidateParameters#ensureBroadcastable(Shape, Shape) broadcastable}
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public Matrix toMatrix(Shape matShape) {
        ValidateParameters.ensureBroadcastable(shape, matShape);
        ValidateParameters.ensureRank(matShape, 2);

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

        if(getRank()==2) {
            mat = new Matrix(shape, entries.clone());
        } else {
            mat = new Matrix(1, entries.length, entries.clone());
        }

        return mat;
    }


    /**
     * Converts this tensor to an equivalent complex tensor.
     * @return A complex tensor equivalent to this real tensor.
     */
    public CTensor toComplex() {
        return new CTensor(shape, ArrayUtils.wrapAsComplex128(entries, null));
    }


    /**
     * Sums this tensor with a dense complex tensor.
     * @param b Dense complex tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CTensor b) {
        return new CTensor(shape, RealComplexDenseOperations.add(b.entries, b.shape, entries, shape));
    }


    /**
     * Sums this tensor with a real sparse tensor.
     * @param b Real sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public Tensor add(CooTensor b) {
        return RealDenseSparseTensorOperations.add(this, b);
    }


    /**
     * Sums this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in sum.
     * @return The element-wise sum of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor add(CooCTensor b) {
        return RealComplexDenseSparseOperations.add(this, b);
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
     * Computes difference of this tensor with a dense complex tensor.
     * @param b Dense complex tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CTensor b) {
        return new CTensor(shape, RealComplexDenseOperations.sub(entries, shape, b.entries, b.shape));
    }


    /**
     * Computes difference of this tensor with a real sparse tensor.
     * @param b Real sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public Tensor sub(CooTensor b) {
        return RealDenseSparseTensorOperations.sub(this, b);
    }


    /**
     * Computes difference of this tensor with a complex sparse tensor.
     * @param b Complex sparse tensor in difference.
     * @return The element-wise difference of this tensor with {@code b}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CTensor sub(CooCTensor b) {
        return RealComplexDenseSparseOperations.sub(this, b);
    }


    /**
     * Subtracts a complex-valued scalar from each entry of this tensor.
     * @param b Scalar to subtract from each entry of this tensor.
     * @return A tensor containing the difference of each entry in this tensor with {@code b}.
     */
    public CTensor sub(Complex128 b) {
        return new CTensor(shape, RealComplexDenseOperations.sub(entries, b));
    }


    /**
     * Computes the element-wise product of this tensor and a complex dense tensor.
     * @param b Complex dense tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CTensor elemMult(CTensor b) {
        return new CTensor(
                this.shape,
                RealComplexDenseElemMult.dispatch(b.entries, b.shape, this.entries, this.shape)
        );
    }


    /**
     * Computes the element-wise product of this tensor and a real sparse tensor.
     * @param b Real sparse tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CooTensor elemMult(CooTensor b) {
        return RealDenseSparseTensorOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of this tensor and a complex sparse tensor.
     * @param b Complex sparse tensor in the element-wise product.
     * @return The element-wise product of this tensor and {@code b}.
     */
    public CooCTensor elemMult(CooCTensor b) {
        return RealComplexDenseSparseOperations.elemMult(this, b);
    }


    /**
     * Computes the element-wise quotient of this tensor and a complex dense tensor.
     * @param b Complex dense tensor in the element-wise quotient.
     * @return The element-wise quotient of this tensor and {@code b}.
     */
    public CTensor elemDiv(CTensor b) {
        return new CTensor(
                shape,
                RealComplexDenseElemDiv.dispatch(entries, shape, b.entries, b.shape)
        );
    }


    /**
     * Computes the scalar multiplication of this tensor and a complex-valued scalar.
     * @param b The complex-valued scalar in the tensor-scalar product.
     * @return The tensor-scalar product of this tensor and {@code b}.
     */
    public CTensor mult(Complex128 b) {
        return new CTensor(shape, Complex128Operations.scalMult(entries, b));
    }


    /**
     * Computes the scalar division of this tensor and a complex-valued scalar.
     * @param b The complex-valued scalar in the tensor-scalar quotient.
     * @return The tensor scalar quotient of this tensor and {@code b}.
     */
    public CTensor div(Complex128 b) {
        return new CTensor(shape, Complex128Operations.scalDiv(entries, b));
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
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
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
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
