/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.core;

import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

import java.io.Serializable;
import java.math.BigInteger;
import java.util.Arrays;

// TODO: Make shape object immutable.
/**
 * An object to store the shape of a tensor. Shapes are immutable.
 */
public class Shape implements Serializable {
    /**
     * An array containing the size of each dimension of this shape.
     */
    private final int[] dims;
    /**
     * An array containing the strides of all dimensions within this shape.
     */
    private int[] strides;
    /**
     * Total entries of this shape.
     */
    private BigInteger totalEntries = null;


    /**
     * Constructs a shape object from specified dimension measurements.
     * @param dims A list of the dimension measurements for this shape object. All entries must be non-negative.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public Shape(int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        ParameterChecks.assertGreaterEq(0, dims);
        this.dims = dims;
    }


    /**
     * Constructs a shape object from specified dimension measurements.
     * @param dims A list of the dimension measurements for this shape object. All entries must be non-negative.
     * @param computeStrides Flag indicating if shape strides should be computed.
     * @throws IllegalArgumentException If any dimension is negative.
     */
    public Shape(boolean computeStrides, int... dims) {
        // Ensure all dimensions for the shape object are non-negative.
        ParameterChecks.assertGreaterEq(0, dims);
        this.dims = dims;
        if(computeStrides) this.strides = this.createNewStrides();
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
        return this.dims.clone();
    }


    /**
     * Gets the shape of a tensor as an array.
     * @return Shape of a tensor as an integer array.
     */
    public int[] getStrides() {
        return this.strides.clone();
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
     * i.e. Strides will be a monotonically decreasing sequence with the last stride being 1.
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
     * If strides are null, create them. Otherwise, do nothing.
     */
    public void makeStridesIfNull() {
        if(strides==null) strides = createNewStrides();
    }


    /**
     * Computes the index of the 1D data array for a dense tensor from tensor indices with this shape.
     * @param indices Indices of tensor with this shape.
     * @return The index of the element at the specified indices in the 1D data array of a dense tensor.
     * @throws IllegalArgumentException If the number of indices does not match the rank of this shape.
     * @throws IndexOutOfBoundsException If any index does not fit within a tensor with this shape.
     */
    public int entriesIndex(int... indices) {
        if(indices.length != dims.length) {
            throw new IllegalArgumentException(ErrorMessages.getIndicesRankErr(indices.length, dims.length));
        }
        if(indices.length>0 && indices[indices.length-1] >= dims[dims.length-1]) {
            throw new IndexOutOfBoundsException("Index " + indices[indices.length-1] + " out of bounds for axis " +
                    (indices.length-1) + " of tensor with shape " + this);
        }

        makeStridesIfNull(); // Computes strides if not previously computed.
        int index = 0;

        for(int i=0; i<indices.length-1; i++) {
            int idx = indices[i];
            if(idx < 0 || idx >= dims[i]) {
                throw new IndexOutOfBoundsException("Index " + idx + " out of bounds for axis " + i +
                        " of tensor with shape " + this);
            }

            index += idx*strides[i];
        }

        return index + indices[indices.length-1];
    }


    /**
     * Computes the nD tensor indices based on an index from the internal 1D data array.
     * @param index Index of internal 1D data array.
     * @return The multidimensional indices corresponding to the 1D data array index. This will be an array of integers
     * with size equal to the rank of this shape.
     */
    public int[] getIndices(int index) {
        int[] indices = new int[this.getRank()];
        indices[indices.length-1] = index % dims[dims.length-1];
        int upStream = index;

        for(int i=indices.length-2; i>=0; i--) {
            upStream = (upStream-indices[i+1]) / dims[i+1];
            indices[i] = upStream%dims[i];
        }

        return indices;
    }


    /**
     * Swaps two axes of this shape. New strides are constructed for this shape.
     * @param axis1 First axis to swap.
     * @param axis2 Second axis to swap.
     * @return A copy of this shape with the specified axis swapped.
     * @throws ArrayIndexOutOfBoundsException If either axis is not within [0, {@link #getRank() rank}-1].
     */
    public Shape swapAxes(int axis1, int axis2) {
        int[] newDims = dims.clone();
        ArrayUtils.swap(newDims, axis1, axis2);
        Shape newShape = new Shape(newDims);

        if(strides!=null) newShape.strides = newShape.createNewStrides();

        return newShape;
    }


    /**
     * Permutes the axes of this shape.
     * @param axes New axes permutation for the shape. This must be a permutation of {@code {1, 2, 3, ... N}} where
     *             {@code N} is the rank of this shape.
     * @return Returns this shape.
     * @throws ArrayIndexOutOfBoundsException If {@code axes} is not a permutation of {@code {1, 2, 3, ... N}}.
     */
    public Shape swapAxes(int... axes) {
        ParameterChecks.assertEquals(getRank(), axes.length);
        ParameterChecks.assertPermutation(axes);

        int[] tempDims = new int[dims.length];

        int i=0;
        for(int axis : axes)  // Permute axes.
            tempDims[i++] = dims[axis];

        Shape newShape = new Shape(tempDims);
        if(strides!=null) newShape.strides = newShape.createNewStrides();

        return newShape;
    }


    /**
     * Gets the total number of entries for a tensor with this shape.
     * @return The total number of entries for a tensor with this shape.
     */
    public BigInteger totalEntries() {
        // Check if totalEntries has already been computed for this shape.
        if(totalEntries!=null) return totalEntries;

        BigInteger product;

        if(dims.length>0) {
            product = BigInteger.ONE;

            for(int dim : dims) {
                product = product.multiply(BigInteger.valueOf(dim));
            }
        } else {
            product = BigInteger.ZERO;
        }

        totalEntries = product;

        return product;
    }


    /**
     * Checks if an object is equal to this shape.
     * @param b Object to compare with this shape.
     * @return True if d is a Shape object and equal to this shape.
     */
    @Override
    public boolean equals(Object b) {
        // Check for early returns.
        if(this == b) return true;
        if(b==null) return false;
        if(b.getClass() != getClass()) return false;

        return Arrays.equals(dims, ((Shape) b).dims);
    }


    /**
     * Gets the next indices for a tensor with this shape.
     * @param currentIndices Current indices. This array is modified.
     * @param i Index of 1d data array.
     */
    public void getNextIndices(int[] currentIndices, int i) {
        for(int j=0; j<currentIndices.length; j++) {
            if((i+1)%strides[j]==0) {
                currentIndices[j] = (currentIndices[j]+1) % dims[j];
            }
        }
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
        StringBuilder result = new StringBuilder("(");

        for(int d : dims)
            result.append(d).append(", ");

        result.replace(result.length()-2, result.length(), ")");  // Remove excess ', ' characters.

        return result.toString();
    }
}
