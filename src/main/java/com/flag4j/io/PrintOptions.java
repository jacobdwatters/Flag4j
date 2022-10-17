package com.flag4j.io;

import com.flag4j.ErrorMessages;

/**
 * Print options for matrices and vectors
 */
public abstract class PrintOptions {

    /**
     * Hide default constructor.
     */
    private PrintOptions() {
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }

    /**
     * Padding between each element of matrix.
     * If negative, zero padding will be used.
     * <br><br>
     *
     * Default Value: 2
     */
    public static int padding = 2;


    /**
     * Maximum number of rows to print from a matrix.
     * If a matrix has more rows than this value, rows with
     * indices larger than this value, except for the last row,
     * will not be printed. If zero or negative all rows will be printed.
     * The last row is always printed<br><br>
     *
     * Default Value: 10
     */
    public static int maxRows = 10;


    /**
     * Maximum number of columns to print from a matrix/vector.
     * If a matrix has more columns than this value, columns with
     * indices equal to or larger than this value, except for the last column,
     * will <b>not</b> be printed. If zero or negative all columns will be printed.
     * The last column is always printed.<br><br>
     *
     * Default Value: 10
     */
    public static int maxColumns = 10;


    /**
     * Precision of the printed matrix values. i.e. the number of decimal places printed.
     * If negative, max precision is used.
     * <br><br>
     * Default Value: 8.
     */
    public static int precision = 8;


    /**
     * A flag for displaying values in scientific notation.
     * <br><br>
     * Default value: false.
     */
    public static boolean useScientific = false;


    /**
     * A flag which indicates if each value should be centered within its column.
     * <br><br>
     * Default value: true.
     */
    public static boolean center = true;
}
