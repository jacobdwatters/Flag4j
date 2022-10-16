package com.flag4j;


/**
 * This interface specifies manipulations which all matrices should implement.
 *
 * @param <T> Matrix type.
 * @param <U> Dense matrix type.
 * @param <V> Sparse matrix type.
 * @param <W> Complex matrix type.
 * @param <Y> Real matrix type.
 * @param <X> Matrix entry type.
 */
interface MatrixManipulationsMixin<T, U, V, W, Y, X extends Number> extends TensorManipulationsMixin<T, U, V, W, Y, X> {

    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     * in the reshaped matrix.
     * @param shape An array of length 2 containing, in order, the number of rows and the number of columns for the
     *              reshaped matrix.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     * @throws IllegalArgumentException If either,<br>
     * - The shape array contains negative indices.<br>
     * - This matrix cannot be reshaped to the specified dimensions.
     */
    T reshape(int[] shape);


    /**
     * Reshapes matrix if possible. The total number of entries in this matrix must match the total number of entries
     *      * in the reshaped matrix.
     * @param numRows The number of rows in the reshaped matrix.
     * @param numCols The number of columns in the reshaped matrix.
     * @return A matrix which is equivalent to this matrix but with the specified dimensions.
     */
    T reshape(int numRows, int numCols);


    /**
     * Flattens a matrix to have a single row. To flatten matrix to a single column, see {@link #flatten(int)}.
     * @return The flat version of this matrix.
     */
    T flatten();


    /**
     * Flattens a matrix along a specified axis. Also see {@link #flatten()}.
     * @param axis - If axis=0, flattens to a row vector.<br>
     *             - If axis=1, flattens to a column vector.
     * @return The flat version of this matrix.
     */
    T flatten(int axis);


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    void setValues(double[][] values);


    /**
     * Sets the value of this matrix using a 2D array.
     * @param values New values of the matrix.
     * @throws IllegalArgumentException If the values array has a different shape then this matrix.
     */
    void setValues(int[][] values);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    void setCol(double[] values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of rows of this matrix.
     */
    void setCol(int[] values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    void setRow(double[] values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values.
     * @param values New values for the row.
     * @param rowIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values array has a different length than the number of columns of this matrix.
     */
    void setRow(int[] values, int rowIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     */
    void setCol(Vector values, int colIndex);


    /**
     * Sets a column of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of rows of this matrix.
     */
    void setCol(SparseVector values, int colIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of columns of this matrix.
     */
    void setRows(Vector values, int rowIndex);


    /**
     * Sets a row of this matrix at the given index to the specified values. Note that the orientation of the values
     * vector is <b>NOT</b> taken into account.
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     * @throws IllegalArgumentException If the values vector has a different length than the number of columns of this matrix.
     */
    void setRows(SparseVector values, int rowIndex);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(double[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(int[][] values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(Matrix values, int rowStart, int colStart);


    /**
     * Sets a slice of this matrix to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    void setSlice(SparseMatrix values, int rowStart, int colStart);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    U setSliceCopy(Matrix values, int rowStart, int colStart);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    T setSliceCopy(SparseMatrix values, int rowStart, int colStart);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    CMatrix setSliceCopy(CMatrix values, int rowStart, int colStart);


    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     * @return A copy of this matrix with the given slice set to the specified values.
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException If the values slice, with upper left corner at the specified location, does not
     * fit completely within this matrix.
     */
    SparseCMatrix setSliceCopy(SparseCMatrix values, int rowStart, int colStart);


    /**
     * Removes a specified row from this matrix.
     * @param rowIndex Index of the row to remove from this matrix.
     */
    void removeRow(int rowIndex);


    /**
     * Removes a specified set of rows from this matrix.
     * @param rowIndices The indices of the rows to remove from this matrix.
     */
    void removeRows(int... rowIndices);


    /**
     * Removes a specified column from this matrix.
     * @param colIndex Index of the column to remove from this matrix.
     */
    void removeCol(int colIndex);


    /**
     * Removes a specified set of columns from this matrix.
     * @param colIndices Indices of the columns to remove from this matrix.
     */
    void removeCols(int... colIndices);


    /**
     * Generates a lower triangular matrix from this matrix. That is, copies all values of this matrix from the principle
     * diagonal and below. The rest of the values in the resultant matrix will be zero.
     * @return A lower triangular matrix whose non-zero values are specified by the values in this matrix at the same indices.
     */
    T tril();


    /**
     * Generates a pseudo-lower triangular matrix from this matrix. That is, copies all values of this matrix from the
     * kth diagonal and below. The rest of the values in the resultant matrix will be zero.
     * @param k The diagonal for which to copy values at and below relative to the principle diagonal.<br>
     *          - If k=0, then all values in and below the principle diagonal will be copied.<br>
     *          - If k=-a, then all values at and below the diagonal which is 'a' diagonals to the left will be copied. <br>
     *          - If k=a, then all values at and below the diagonal which is 'a' diagonals to the right will be copied.
     * @return A lower triangular matrix whose non-zero values are specified by the values in this matrix at the same indices.
     * @throws IllegalArgumentException If k is out of the range of this matrix. That is, if k specifies a diagonal which
     * does not exist in this matrix.
     */
    T tril(int k);


    /**
     * Generates an upper triangular matrix from this matrix. That is, copies all values of this matrix from the principle
     * diagonal and above. The rest of the values in the resultant matrix will be zero.
     * @return An upper triangular matrix whose non-zero values are specified by the values in this matrix at the same indices.
     */
    T triu();


    /**
     * Generates a pseudo-upper triangular matrix from this matrix. That is, copies all values of this matrix from the
     * kth diagonal and above. The rest of the values in the resultant matrix will be zero.
     * @param k The diagonal for which to copy values at and below relative to the principle diagonal.<br>
     *          - If k=0, then all values in and above the principle diagonal will be copied.<br>
     *          - If k=-a, then all values at and above the diagonal which is 'a' diagonals to the left will be copied. <br>
     *          - If k=a, then all values at and above the diagonal which is 'a' diagonals to the right will be copied.
     * @return An upper triangular matrix whose non-zero values are specified by the values in this matrix at the same indices.
     * @throws IllegalArgumentException If k is out of the range of this matrix. That is, if k specifies a diagonal which
     * does not exist in this matrix.
     */
    T triu(int k);


    /**
     * Generates a diagonal matrix from this matrix. This is, copies all values of this matrix from the principle diagonal. The
     * rest of the values in the resultant matrix will be zero.
     * @return A diagonal matrix whose non-zero entries are specified by the values in this matrix.
     */
    T diag();


    /**
     * Generates a diagonal matrix from this matrix. This is, copies all values of this matrix from the principle diagonal
     * and the k diagonals to the left and k diagonals to the right.
     * The
     * rest of the values in the resultant matrix will be zero.
     * @param k The number of diagonals in each direction (left or right) to include in the resultant matrix.
     * @return A diagonal matrix whose non-zero entries are specified by the values in this matrix.
     * @throws IllegalArgumentException - If k is negative.<br>
     * - If k is out of range of this matrix.
     */
    T diag(int k);


    /**
     * Rounds this matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     * @return A copy of this matrix with each entry rounded to the nearest whole number.
     */
    T round();


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    T round(int precision);


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within 1.0*E^-12 of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this matrix with rounded values.
     */
    T roundToZero();


    /**
     * Rounds values which are close to zero in absolute value to zero. If the matrix is complex, both the real and imaginary components will be rounded
     * independently.
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    T roundToZero(double threshold);
}
