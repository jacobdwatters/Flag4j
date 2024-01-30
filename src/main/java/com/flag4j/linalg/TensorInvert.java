package com.flag4j.linalg;

import com.flag4j.*;
import com.flag4j.core.TensorBase;
import com.flag4j.core.TensorExclusiveMixin;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ParameterChecks;


/**
 * This class provides methods for computing the 'inverse' of a tensor with respect to some tensor dot product operation.
 */
public final class TensorInvert {

    private TensorInvert() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src Tensor to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    public static Tensor inv(Tensor src, int numIndices) {
        ParameterChecks.assertPositive(numIndices);

        Shape originalShape = src.shape.copy();
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.dims, numIndices);

        // Convert to an equivalent matrix inverse problem.
        Matrix matInverse = new Matrix(prod, src.entries.length-prod, src.entries).inv();

        return new Tensor(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * <p>Computes the 'inverse' of this tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link TensorExclusiveMixin#tensorDot(TensorBase, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src Tensor to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    public static CTensor inv(CTensor src, int numIndices) {
        ParameterChecks.assertPositive(numIndices);

        Shape originalShape = src.shape.copy();
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.dims, numIndices);

        // Convert to an equivalent matrix inverse problem.
        CMatrix matInverse = new CMatrix(prod, src.entries.length-prod, src.entries).inv();

        return new CTensor(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * Computes the shape of the 'inverse' tensor given the original shape and the number of indices.
     * @param originalShape Shape of the original tensor.
     * @param numIndices Number of indices that are involved in the inverse sum.
     * @return The shape of the 'inverse' tensor.
     */
    private static Shape getInvShape(Shape originalShape, int numIndices) {
        int[] invDims = new int[2*numIndices];
        System.arraycopy(originalShape.dims, numIndices, invDims, 0, numIndices);
        System.arraycopy(originalShape.dims, 0, invDims, numIndices, 2 * numIndices - numIndices);
        return new Shape(true, invDims);
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
