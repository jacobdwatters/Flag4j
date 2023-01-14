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

package com.flag4j.io;

import com.flag4j.util.ErrorMessages;

/**
 * Print options for matrices and vectors
 */
public abstract class PrintOptions {

    /**
     * Default padding between elements when printing.
     */
    public static final int DEFAULT_PADDING = 2;
    /**
     * Default maximum number of rows to print.
     */
    public static final int DEFAULT_MAX_ROWS = 10;
    /**
     * Default maximum number of columns to print.
     */
    public static final int DEFAULT_MAX_COLS = 10;
    /**
     * Default precision (i.e. number of decimals) to use when printing.
     */
    public static final int DEFAULT_PRECISION = 8;
    /**
     * Default flag for centering elements when printing.
     */
    public static final boolean DEFAULT_CENTER = true;

    /**
     * Hide default constructor.
     */
    private PrintOptions() {
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }

    /**
     * Padding between each element of matrix.
     * If negative, zero padding will be used.
     * <br><br>
     *
     * Default Value: 2
     */
    private static int padding = DEFAULT_PADDING;


    /**
     * Maximum number of rows to print from a matrix.
     * If a matrix has more rows than this value, rows with
     * indices larger than this value, except for the last row,
     * will not be printed. If zero or negative all rows will be printed.
     * The last row is always printed<br><br>
     *
     * Default Value: 10
     */
    private static int maxRows = DEFAULT_MAX_ROWS;


    /**
     * Maximum number of columns to print from a matrix/vector.
     * If a matrix has more columns than this value, columns with
     * indices equal to or larger than this value, except for the last column,
     * will <b>not</b> be printed. If zero or negative all columns will be printed.
     * The last column is always printed.<br><br>
     *
     * Default Value: 10
     */
    private static int maxColumns = DEFAULT_MAX_COLS;


    /**
     * Precision of the printed matrix values. i.e. the number of decimal places printed.
     * If negative, max precision is used.
     * <br><br>
     * Default Value: 8.
     */
    private static int precision = DEFAULT_PRECISION;


    /**
     * A flag which indicates if each value should be centered within its column.
     * <br><br>
     * Default value: true.
     */
    private static boolean center = DEFAULT_CENTER;


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
     * Gets the current maximum number of columns to print.
     * @return The current maximum number of columns to print.
     */
    public static int getMaxColumns() {
        return maxColumns;
    }


    /**
     * Sets the maximum number of columns to print.
     * @param maxColumns Maximum number of columns to print.
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


    /**
     * Resets all print options to their default values.
     */
    public static void resetAll() {
        padding = DEFAULT_PADDING;
        maxColumns = DEFAULT_MAX_COLS;
        maxRows = DEFAULT_MAX_ROWS;
        precision = DEFAULT_PRECISION;
        center = DEFAULT_CENTER;
    }
}
