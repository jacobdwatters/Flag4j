package com.flag4j.operations;

import com.flag4j.core.TensorBase;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

/**
 * This class provides methods for computing operations on real dense tensors.
 */
public final class RealDenseOperations {

    private RealDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param A First Tensor of the addition.
     * @param A Second Tensor of the addition.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] add(TensorBase<double[]> A, TensorBase<double[]> B) {
        ShapeChecks.equalShapeCheck(A.shape, B.shape);

        double[] sum = new double[A.entries.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = A.entries[i] + B.entries[i];
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param entries1 Entries of first tensor.
     * @param entries2 Entries of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] sub(double[] entries1, double[] entries2) {
        ShapeChecks.arrayLengthsCheck(entries1.length, entries2.length);

        double[] sum = new double[entries1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = entries1[i] - entries2[i];
        }

        return sum;
    }


    /**
     * Sums all entries in a tensor.
     * @param entries The entries of the tensor.
     * @return The sum of all entries in the tensor.
     */
    public static double sum(double[] entries) {
        double sum = 0;

        for(int i=0; i<entries.length; i++) {
            sum += entries[i];
        }

        return sum;
    }


    /**
     * Multiplies all entries in a tensor.
     * @param entries The entries of the tensor.
     * @return The product of all entries in the tensor.
     */
    public static double prod(double[] entries) {
        double product;

        if(entries.length > 0) {
            product=1;
            for(int i=0; i<entries.length; i++) {
                product *= entries[i];
            }
        } else {
            product=0;
        }

        return product;
    }
}
