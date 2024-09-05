/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.core_old.TensorBase;
import org.flag4j.core_old.TensorExclusiveMixin;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;


/**
 * This utility class provides methods for computing the 'inverse' of a tensor with respect to some tensor dot product operation.
 */
public final class TensorInvert {

    private TensorInvert() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src TensorOld to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    @Deprecated
    public static TensorOld inv(TensorOld src, int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem and solve.
        MatrixOld matInverse = InvertOld.inv(new MatrixOld(prod, src.entries.length-prod, src.entries));

        return new TensorOld(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     *
     * @param src TensorOld to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    public static <T extends PrimitiveDoubleTensorBase<T, T>> T inv(PrimitiveDoubleTensorBase<T, T> src, int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem and solve.
        MatrixOld matInverse = InvertOld.inv(new MatrixOld(prod, src.entries.length-prod, src.entries));

        return src.makeLikeTensor(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src TensorOld to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    @Deprecated
    public static CTensorOld inv(CTensorOld src, int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem.
        CMatrixOld matInverse = InvertOld.inv(new CMatrixOld(prod, src.entries.length-prod, src.entries));

        return new CTensorOld(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * Computes the shape of the 'inverse' tensor given the original shape and the number of indices.
     * @param originalShape Shape of the original tensor.
     * @param numIndices Number of indices that are involved in the inverse sum.
     * @return The shape of the 'inverse' tensor.
     */
    private static Shape getInvShape(Shape originalShape, int numIndices) {
        int[] invDims = new int[2*numIndices];
        int[] origDims = originalShape.getDims();
        System.arraycopy(origDims, numIndices, invDims, 0, numIndices);
        System.arraycopy(origDims, 0, invDims, numIndices, 2 * numIndices - numIndices);
        return new Shape(invDims);
    }


    /**
     * Computes the total number of entries in the last {@code numIndices} dimensions of a tensor with dimensions specified by
     * {@code dims}.
     * @param dims Dimensions of the tensor.
     * @param numIndices Number of last indices to compute product for.
     * @return The total number of entries in the last {@code numIndices} dimensions of a tensor with dimensions specified by
     * {@code dims}. That is, the product of the last {@code numIndices} entries of {@code dims}.
     */
    private static int getProduct(int[] dims, int numIndices) {
        int prod = 1;
        for(int k=numIndices; k<dims.length; k++) {
            prod *= dims[k];
        }

        return prod;
    }
}
