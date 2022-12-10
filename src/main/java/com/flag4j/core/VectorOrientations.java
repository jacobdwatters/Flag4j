package com.flag4j.core;

import com.flag4j.util.ErrorMessages;

/**
 * A simple enum class which contains possible orientations for a vector. i.e. a row or column vector.
 */
public enum VectorOrientations {

    /**
     * Indicates row vector.
     */
    ROW,
    /**
     * Indicates column vector.
     */
    COL,
    /**
     * Indicates that the vector is not oriented. It will be treated as a row/column vector as needed.
     */
    UNORIENTED;


    public static VectorOrientations getFromOrdinal(int ordinal) {
        if(ordinal==ROW.ordinal()) {
            return ROW;
        } else if(ordinal==COL.ordinal()) {
            return COL;
        } else if(ordinal==UNORIENTED.ordinal()) {
            return UNORIENTED;
        } else {
            throw new IllegalArgumentException(ErrorMessages.vectorOrientationErr(ordinal));
        }
    }
}
