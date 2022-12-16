/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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

package com.flag4j;

import com.flag4j.util.ErrorMessages;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;

/**
 * An object to store the shape of a tensor. Note that this object is mutable.
 */
public class Shape implements Serializable {
    /**
     * An array containing the size of each dimension of this shape.
     */
    public int[] dims;
    /**
     * An array containing the strides of all dimensions within this shape.
     */
    public int[] strides;


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
        this.strides = this.createNewStrides();
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
     * Gets the shape of a tensor as an array.
     * @return Shape of a tensor as an integer array.
     */
    public int[] getStrides() {
        return this.strides;
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
     * Constructs strides for each dimension of this shape as if for a newly constructed tensor.
     * i.e. Strides will be a decreasing sequence with the last stride being 1.
     * @return The strides for all dimensions of a newly constructed tensor with this shape.
     */
    public int[] createNewStrides() {
        int[] strides = new int[dims.length];

        if(strides.length>0) {
            // Stride along last axis is always one for new strides.
            strides[strides.length-1] = 1;

            for(int i=strides.length-2; i>=0; i--) {
                strides[i] = dims[i+1]*strides[i+1];
            }
        }

        return strides;
    }


    /**
     * Computes the index of the 1D data array for a dense tensor from tensor indices with this shape.
     * @param indices Indices of tensor with this shape.
     * @return The index of the element at the specified indices in the 1D data array of a dense tensor.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this shape.
     * @throws ArrayIndexOutOfBoundsException If any index does not fit within a tensor with this shape.
     */
    public int entriesIndex(int... indices) {
        if(indices.length != dims.length) {
            throw new IllegalArgumentException(ErrorMessages.getIndicesRankErr(indices.length, dims.length));
        }
        if(indices.length>0 && indices[indices.length-1] >= dims[dims.length-1]) {
            throw new ArrayIndexOutOfBoundsException("Index " + indices[indices.length-1] + " out of bounds for axis " +
                    (indices.length-1) + " of tensor with shape " + this);
        }

        int index = 0;

        for(int i=0; i<indices.length-1; i++) {
            if(indices[i] < 0 || indices[i] >= dims[i]) {
                throw new ArrayIndexOutOfBoundsException("Index " + indices[i] + " out of bounds for axis " + i +
                        " of tensor with shape " + this);
            }

            index += indices[i]*strides[i];
        }

        return index + indices[indices.length-1];
    }


    /**
     * Swaps two axes of this shape. New strides are constructed for this shape.
     * @param axis1 First axis to swap.
     * @param axis2 Second axis to swap.
     * @return Returns this shape.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within [0, {@link #getRank() rank}-1].
     */
    public Shape swapAxes(int axis1, int axis2) {
        int temp = dims[axis1];
        dims[axis1] = dims[axis2];
        dims[axis2] = temp;

        this.strides = this.createNewStrides();

        return this;
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
     * Creates a deep copy of this shape object. This is a distinct object not a reference to the same object.
     * @return A deep copy of this shape object.
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

        // Ensure the object is the same type
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
            result.append(d).append("x");
        }
        result.deleteCharAt(result.length()-1); // Remove excess 'x' character.

        return result.toString();
    }
}
