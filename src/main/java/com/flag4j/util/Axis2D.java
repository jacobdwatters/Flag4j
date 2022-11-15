package com.flag4j.util;

/**
 * Simple enum class for two-dimensional axis. The row axis has ordinal 0 and the column axis has ordinal 1.
 */
public enum Axis2D {
    ROW, COL;

    /**
     * Get the ordinal of the row axis in 2D.
     * @return The ordinal of the row axis in 2D.
     */
    public static int row() {
        return ROW.ordinal();
    }


    /**
     * Get the ordinal of the column axis in 2D.
     * @return The ordinal of the column axis in 2D.
     */
    public static int col() {
        return COL.ordinal();
    }


    /**
     * Gets an array of all axis ordinals.
     * @return An array of all axis ordinals.
     */
    public static int[] allAxes() {
        return new int[]{ROW.ordinal(), COL.ordinal()};
    }
}
