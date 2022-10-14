package com.flag4j;

/**
 * An object to store the shape of a tensor. Note that the array holding the shape of the tensor is mutable.
 */
public class Shape {
    public int[] shape;


    /**
     * Constructs a shape object with a specified rank such that all dimensions measure zero.
     * @param rank Rank of the tensor which this shape object describes.
     */
    public Shape(int rank) {
        shape = new int[rank];
    }


    /**
     * Constructs a shape object from specified dimension measurements.
     * @param dims A list of the dimension measurements for this shape object.
     */
    public Shape(int... dims) {
        this.shape = dims.clone();
    }


    /**
     * Gets the rank of a tensor with this shape.
     * @return The rank for a tensor with this shape.
     */
    public int getRank() {
        return shape.length;
    }


    /**
     * Converts this Shape object to a string format.
     * @return The string representation for this Shape object.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("");

        for(int d : shape) {
            result.append(d + "x");
        }
        result.deleteCharAt(result.length()-1); // Remove excess 'x' character.

        return result.toString();
    }
}
