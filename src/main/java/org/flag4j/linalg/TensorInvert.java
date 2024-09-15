package org.flag4j.linalg;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays.backend.TensorOverSemiRing;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This utility class provides static methods for computing the 'inverse' of a tensor with respect to some
 * tensor dot product operation.
 */
public final class TensorInvert {

    private TensorInvert() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link Tensor#tensorDot(TensorOverSemiRing, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src Tensor to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    @Deprecated
    public static Tensor inv(Tensor src, int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem and solve.
        Matrix matInverse = Invert.inv(new Matrix(prod, src.entries.length-prod, src.entries));

        return new Tensor(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link Tensor#tensorDot(TensorOverSemiRing, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the
     * tensor dot product operation.</p>
     *
     * <p>A tensor {@code I} is the identity for a tensor dot product if
     * {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     *
     * @param src Tensor to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    public static <T extends PrimitiveDoubleTensorBase<T, T>> T inv(PrimitiveDoubleTensorBase<T, T> src,
                                                                    int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem and solve.
        Matrix matInverse = Invert.inv(new Matrix(prod, src.entries.length-prod, src.entries));

        return src.makeLikeTensor(invShape, matInverse.entries); // Reshape as tensor.
    }


    /**
     * <p>Computes the 'inverse' of a tensor. That is, computes the tensor {@code X=this.tensorInv()} such that
     * {@link Tensor#tensorDot(TensorOverSemiRing, int) src.tensorDot(X, numIndices)} is the 'identity' tensor for the tensor
     * dot product operation.</p>
     * <p>A tensor {@code I} is the identity for a tensor dot product if {@code src.tensorDot(I, numIndices).equals(this)}.</p>
     * @param src Tensor to compute inverse of.
     * @param numIndices The number of first numIndices which are involved in the inverse sum.
     * @return The 'inverse' of this tensor as defined in the above sense.
     * @throws IllegalArgumentException If {@code numIndices} is not positive.
     */
    @Deprecated
    public static CTensor inv(CTensor src, int numIndices) {
        ParameterChecks.ensurePositive(numIndices);

        Shape originalShape = src.shape;
        Shape invShape = getInvShape(originalShape, numIndices);
        int prod = getProduct(originalShape.getDims(), numIndices);

        // Convert to an equivalent matrix inverse problem.
        CMatrix matInverse = Invert.inv(new CMatrix(prod, src.entries.length-prod, src.entries));

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
        
        for(int k=numIndices; k<dims.length; k++)
            prod *= dims[k];

        return prod;
    }
}
