package org.flag4j.linalg;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.backend.FieldTensorBase;
import org.flag4j.arrays.backend.PrimitiveDoubleTensorBase;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.operations.common.field_ops.CompareField;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * This utility class provides static methods useful for computing norms of a tensor.
 */
public final class TensorNorms {

    private TensorNorms() {
        // Hide default constructor for utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }

    // TODO: Ensure the below infNorm methods correct? These seems to be a max norm. Are the same?
    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest absolute value.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source tensor, matrix, or vector.
     */
    public static double infNorm(PrimitiveDoubleTensorBase<?, ?> src) {
        return src.maxAbs();
    }


    /**
     * Computes the infinity norm of a tensor, matrix, or vector. That is, the largest value by magnitude.
     * @param src The tensor, matrix, or vector to compute the norm of.
     * @return The infinity norm of the source tensor, matrix, or vector.
     */
    public static double infNorm(FieldTensorBase<?, ?, ?> src) {
        return src.maxAbs();
    }


    /**
     * Computes the 2-norm of this tensor as if the tensor was a vector (i.e. as if by {@code VectorNorm(Tensor.toVector())}).
     * This is equivalent to {@link #norm(Tensor, double) norm(src, 2)}.
     *
     * @param src Tensor to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(Tensor src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Tensor to compute norm of.
     * @param p The {@code p} value in the p-norm. <br>
     *          - If {@code p} is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public static double norm(Tensor src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CTensor, double) norm(src, 2)}.
     *
     * @return the 2-norm of this tensor.
     */
    public double norm(CTensor src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Tensor to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CTensor src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the maximum/infinite norm of this tensor.
     *
     * @param src Tensor to compute norm of.
     * @return The maximum/infinite norm of this tensor.
     */
    public double infNorm(CTensor src) {
        return CompareField.maxAbs(src.entries);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooTensor, double) norm(src, 2)}.
     *
     * @param src Tensor to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooTensor src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Tensor to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CooTensor src, double p) {
        return tensorNormLp(src.entries, p);
    }


    /**
     * Computes the 2-norm of this tensor. This is equivalent to {@link #norm(CooTensor, double) norm(src, 2)}.
     *
     * @param src Tensor to compute norm of.
     * @return the 2-norm of this tensor.
     */
    public static double norm(CooCTensor src) {
        return tensorNormL2(src.entries);
    }


    /**
     * Computes the p-norm of this tensor.
     *
     * @param src Tensor to compute norm of.
     * @param p The p value in the p-norm. <br>
     *          - If p is inf, then this method computes the maximum/infinite norm.
     * @return The p-norm of this tensor.
     * @throws IllegalArgumentException If p is less than 1.
     */
    public double norm(CooCTensor src, double p) {
        return tensorNormLp(src.entries, p);
    }


    // ---------------------------- Low-level implementations ----------------------------

    /**
     * Computes the L<sub>2</sub> norm of a tensor.
     * @param src Entries of the tensor.
     * @return The L<sub>2</sub> norm of the tensor.
     */
    public static double tensorNormL2(double[] src) {
        double norm = 0;

        for(double value : src)
            norm += Math.pow(Math.abs(value), 2);

        return Math.sqrt(norm);
    }


    /**
     * Computes the L<sub>p</sub> norm of a tensor.
     * @param src Entries of the tensor.
     * @param p The {@code p} parameter of the L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the tensor.
     */
    public static double tensorNormLp(double[] src, double p) {
        ParameterChecks.ensureNotEquals(0, p);
        double norm = 0;

        for(double value : src)
            norm += Math.pow(Math.abs(value), p);

        return Math.pow(norm, 1.0/p);
    }


    /**
     * Computes the L<sub>2</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @return The L<sub>2</sub> norm of the tensor.
     */
    public static double tensorNormL2(Complex128[] src) {
        double norm = 0;

        for(Complex128 cNumber : src)
            norm += Complex128.pow(cNumber, 2).mag();

        return Math.sqrt(norm);
    }


    /**
     * Computes the L<sub>p</sub> norm of a tensor (i.e. the Frobenius norm).
     * @param src Entries of the tensor.
     * @param p The {@code p} parameter of the L<sub>p</sub> norm.
     * @return The L<sub>p</sub> norm of the tensor.
     */
    public static double tensorNormLp(Complex128[] src, double p) {
        double norm = 0;

        for(Complex128 cNumber : src)
            norm += Complex128.pow(cNumber, p).mag();

        return Math.pow(norm, 1.0/p);
    }
}
