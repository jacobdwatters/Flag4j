package com.flag4j.io;

import com.flag4j.util.ErrorMessages;

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
    private static int padding = 2;


    /**
     * Maximum number of rows to print from a matrix.
     * If a matrix has more rows than this value, rows with
     * indices larger than this value, except for the last row,
     * will not be printed. If zero or negative all rows will be printed.
     * The last row is always printed<br><br>
     *
     * Default Value: 10
     */
    private static int maxRows = 10;


    /**
     * Maximum number of columns to print from a matrix/vector.
     * If a matrix has more columns than this value, columns with
     * indices equal to or larger than this value, except for the last column,
     * will <b>not</b> be printed. If zero or negative all columns will be printed.
     * The last column is always printed.<br><br>
     *
     * Default Value: 10
     */
    private static int maxColumns = 10;


    /**
     * Precision of the printed matrix values. i.e. the number of decimal places printed.
     * If negative, max precision is used.
     * <br><br>
     * Default Value: 8.
     */
    private static int precision = 8;


    /**
     * A flag for displaying values in scientific notation.
     * <br><br>
     * Default value: false.
     */
    private static boolean useScientific = false;


    /**
     * A flag which indicates if each value should be centered within its column.
     * <br><br>
     * Default value: true.
     */
    private static boolean center = true;


    /**
     * Sets the centering flag.
     * @param center Flag for centering values within its column.
     */
    public static void setCenter(boolean center) {
        PrintOptions.center = center;
    }


    /**
     * Gets the centering flag.
     * @return The current value of the center flag.
     */
    public static boolean getCenter() {
        return center;
    }


    /**
     * Sets the printing precision for which values in a matrix/vector
     * @param precision The precision to use.
     */
    public static void setPrecision(int precision) {
        if(precision < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(precision));
        }

        PrintOptions.precision = precision;
    }


    /**
     * Gets the current printing precision.
     * @return The current printing precision.
     */
    public static int getPrecision() {
        return precision;
    }


    /**
     * Sets the flag for printing in scientific notation.
     * @param useScientific Flag to use for printing scientific notation.
     */
    public static void setUseScientific(boolean useScientific) {
        PrintOptions.useScientific = useScientific;
    }


    /**
     * Gets the current flag for printing in scientific notation.
     * @return The current flag for weather to print values in scientific notation or not.
     */
    public static boolean getUseScientific() {
        return useScientific;
    }


    /**
     * Gets the current maximum number of columns to print.
     * @return The current maximum number of columns to print.
     */
    public static int getMaxColumns() {
        return maxColumns;
    }


    /**
     * Sets the maximum number of columns to print.
     * @param maxColumns
     */
    public static void setMaxColumns(int maxColumns) {
        if(maxColumns < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(maxColumns));
        }

        PrintOptions.maxColumns = maxColumns;
    }


    /**
     * Gets the maximum number of rows to print.
     * @return The maximum number of rows to print.
     */
    public static int getMaxRows() {
        return maxRows;
    }


    /**
     * Set the maximum number of rows to print.
     * @param maxRows The new maximum number of rows to print.
     */
    public static void setMaxRows(int maxRows) {
        if(maxRows < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(maxRows));
        }

        PrintOptions.maxRows = maxRows;
    }


    /**
     * Gets the current padding amount for columns.
     * @return The current padding amount for columns.
     */
    public static int getPadding() {
        return padding;
    }


    /**
     * Sets the padding amount for the columns.
     * @param padding New padding amount for the columns.
     */
    public static void setPadding(int padding) {
        if(padding < 0) {
            throw new IllegalArgumentException(ErrorMessages.negValueErr(padding));
        }

        PrintOptions.padding = padding;
    }
}
