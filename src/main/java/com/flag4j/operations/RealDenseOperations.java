package com.flag4j.operations;

import com.flag4j.Shape;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeChecks;

/**
 * This class provides low level methods for computing operations on real dense tensors.
 */
public final class RealDenseOperations {

    private RealDenseOperations() {
        // Hide constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the element-wise addition of two tensors.
     * @param entries1 Entries of first Tensor of the addition.
     * @param shape1 Shape of first tensor.
     * @param entries2 Entries of second Tensor of the addition.
     * @param shape2 Shape of second tensor.
     * @return The element wise addition of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] add(double[] entries1, Shape shape1, double[] entries2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);
        double[] sum = new double[entries1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = entries1[i] + entries2[i];
        }

        return sum;
    }


    /**
     * Adds a scalar to every element of a tensor.
     * @param entries entries of tensor to add scalar to.
     * @param b Scalar to add to tensor.
     * @return The tensor scalar addition.
     */
    public static double[] add(double[] entries, double b) {
        double[] sum = new double[entries.length];

        for(int i=0; i<entries.length; i++) {
            sum[i] = entries[i] + b;
        }

        return sum;
    }


    /**
     * Computes the element-wise subtraction of two tensors.
     * @param entries1 Entries of first tensor.
     * @param shape1 Shape of first tensor.
     * @param entries2 Entries of second tensor.
     * @param shape2 Shape of second tensor.
     * @return The element wise subtraction of two tensors.
     * @throws IllegalArgumentException If entry arrays are not the same size.
     */
    public static double[] sub(double[] entries1, Shape shape1, double[] entries2, Shape shape2) {
        ShapeChecks.equalShapeCheck(shape1, shape2);
        double[] sum = new double[entries1.length];

        for(int i=0; i<sum.length; i++) {
            sum[i] = entries1[i] - entries2[i];
        }

        return sum;
    }


    /**
     * Sums all entries in a tensor.
     * @param entries Entries of tensor so sum.
     * @return The sum of all entries in the tensor.
     */
    public static double sum(double[] entries) {
        double sum = 0;

        for(double value : entries) {
            sum += value;
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
            for(double value : entries) {
                product *= value;
            }
        } else {
            product=0;
        }

        return product;
    }
}
