package com.flag4j;

import com.flag4j.util.ErrorMessages;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;

/**
 * An object to store the shape of a tensor. Note that the array holding the shape of the tensor is mutable.
 */
public class Shape implements Serializable {
    /**
     * An array containing the size of each dimension of this tensor.
     */
    public int[] dims;


    /**
     * Constructs a shape object from specified dimension measurements.
     * @param dims A list of the dimension measurements for this shape object. All entries must be non-negative.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public Shape(int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        if(Arrays.stream(dims).anyMatch(i -> i < 0)) {
            throw new IllegalArgumentException(ErrorMessages.negativeDimErrMsg(dims));
        }

        this.dims = dims.clone();
    }


    /**
     * Gets the rank of a tensor with this shape.
     * @return The rank for a tensor with this shape.
     */
    public int getRank() {
        return dims.length;
    }


    /**
     * Gets the shape of a tensor as an array.
     * @return Shape of a tensor as an integer array.
     */
    public int[] getDims() {
        return this.dims;
    }


    /**
     * Get the size of the shape object in the specified dimension.
     * @param i Dimension to get the size of.
     * @return The size of this shape object in the specified dimension.
     */
    public int get(int i) {
        return this.dims[i];
    }


    /**
     * Gets the total number of entries for a tensor with this shape.
     * @return The total number of entries for a tensor with this shape.
     */
    public BigInteger totalEntries() {
        BigInteger product;

        if(dims.length>0) {
            product = BigInteger.ONE;

            for(int dim : dims) {
                product = product.multiply(BigInteger.valueOf(dim));
            }
        } else {
            product = BigInteger.ZERO;
        }

        return product;
    }


    /**
     * Creates a copy of this shape object.
     * @return
     */
    @Override
    public Shape clone() {
        return new Shape(dims.clone());
    }


    /**
     * Checks if an object is equal to this shape.
     * @param b Object to compare with this shape.
     * @return True if d is a Shape object and equal to this shape.
     */
    @Override
    public boolean equals(Object b) {
        boolean result = true;

        if(b instanceof Shape) {
            Shape bCopy = (Shape) b;

            if(this.dims.length == bCopy.dims.length) {
                for(int i=0; i<dims.length; i++) {
                    if(dims[i] != bCopy.dims[i]) {
                        result = false;
                        break;
                    }
                }
            } else {
                result = false;
            }

        } else {
            result = false;
        }

        return result;
    }


    /**
     * Generates the hashcode for this shape object. This is computed by passing the dims array of this shape object to
     * {@link java.util.Arrays#hashCode(int[])}.
     * @return The hashcode for this array object.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(this.dims);
    }


    /**
     * Converts this Shape object to a string format.
     * @return The string representation for this Shape object.
     */
    public String toString() {
        StringBuilder result = new StringBuilder();

        for(int d : dims) {
            result.append(d + "x");
        }
        result.deleteCharAt(result.length()-1); // Remove excess 'x' character.

        return result.toString();
    }
}
