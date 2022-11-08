package com.flag4j;

/**
 * An object to store the shape of a tensor. Note that the array holding the shape of the tensor is mutable.
 */
public class Shape {
    /**
     * An array containing the size of each dimension of this tensor.
     */
    public int[] dims;


    /**
     * Constructs a shape object with a specified rank such that all dimensions measure zero.
     * @param rank Rank of the tensor which this shape object describes.
     */
    public Shape(int rank) {
        dims = new int[rank];
    }


    /**
     * Constructs a shape object from specified dimension measurements.
     * @param dims A list of the dimension measurements for this shape object.
     */
    public Shape(int... dims) {
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
     * Converts this Shape object to a string format.
     * @return The string representation for this Shape object.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("");

        for(int d : dims) {
            result.append(d + "x");
        }
        result.deleteCharAt(result.length()-1); // Remove excess 'x' character.

        return result.toString();
    }
}
