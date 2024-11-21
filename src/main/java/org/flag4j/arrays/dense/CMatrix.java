/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.field.AbstractDenseFieldMatrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex dense matrix backed by a {@link Complex128} array.
 *
 * <p>A CMatrix has mutable entries but fixed shape.
 *
 * <p>A matrix is essentially equivalent to a rank 2 tensor but has some extended functionality and <i>may</i>
 * have improved performance for some operations.
 */
public class CMatrix extends AbstractDenseFieldMatrix<CMatrix, CVector, Complex128> {

    // TODO: Add isReal() and isComplex() methods also in the above mentioned complex tensor classes.

    /**
     * Creates a complex matrix with the specified {@code entries} and {@code shape}.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(Shape shape, Field<Complex128>[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code shape} filled with {@code fillValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a zero matrix with the specified {@code shape}.
     *
     * @param shape Shape of this matrix.
     */
    public CMatrix(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a square zero matrix with the specified {@code size}.
     *
     * @param size Size of the zero matrix to construct. The resulting matrix will have shape {@code (size, size)}
     */
    public CMatrix(int size) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code entries}, and shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(int rows, int cols, Field<Complex128>[] entries) {
        super(new Shape(rows, cols), entries);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified shape and filled with {@code fillValue}.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(int rows, int cols, Complex128 fillValue) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a zero matrix with the specified shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     */
    public CMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Constructs a complex matrix from a 2D array. The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(Field<Complex128>[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(Field<Complex128>[] row : entries) {
            for(Field<Complex128> value : row)
                super.entries[flatPos++] = value;
        }
    }


    /**
     * <p>Constructs a complex matrix from a 2D array of strings. Each string must be formatted properly as a complex number that can
     * be parsed by {@link org.flag4j.io.parsing.ComplexNumberParser}</p>
     *
     * <p>The matrix will have the same shape as the array.</p>
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(String[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(String[] row : entries) {
            for(String value : row)
                super.entries[flatPos++] = new Complex128(value);
        }
    }


    /**
     * Constructs a complex matrix with specified {@code shape} and {@code entries}.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix.
     */
    public CMatrix(Shape shape, double[] entries) {
        super(shape, new Complex128[entries.length]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
        ArrayUtils.arraycopy(entries, 0, super.entries, 0, entries.length);
    }


    /**
     * Constructs a complex matrix from a 2D array of double values.
     * @param aEntriesReal Entries of the complex matrix to construct. Each value will be wrapped as {@link Complex128 Complex128's}.
     */
    public CMatrix(double[][] aEntriesReal) {
        super(new Shape(aEntriesReal.length, aEntriesReal[0].length), new Complex128[aEntriesReal.length*aEntriesReal[0].length]);
        setZeroElement(Complex128.ZERO);

        int idx = 0;
        for(double[] row : aEntriesReal) {
            for(double value : row)
                super.entries[idx++] = new Complex128(value);
        }
    }


    /**
     * Constructs a matrix with the specified shape filled with {@code fillValue}.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of rows in the matrix.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int numRows, int numCols, double fillValue) {
        super(new Shape(numRows, numCols), new Complex128[numRows*numCols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Complex128 fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Double fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates matrix with the specified {@code shape} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(Shape shape, Double fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param mat Matrix to create copy of.
     */
    public CMatrix(CMatrix mat) {
        super(mat.shape, mat.entries.clone());
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected CVector makeLikeVector(Field<Complex128>[] entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a sparse COO matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     *
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    @Override
    protected CooCMatrix makeLikeCooMatrix(Shape shape, Field<Complex128>[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse CSR matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero entries of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    @Override
    protected CsrCMatrix makeLikeCsrMatrix(Shape shape, Field<Complex128>[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero entries of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCMatrix makeLikeCooTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        return makeLikeCooMatrix(shape, entries, indices[0], indices[1]);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code entries}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code entries}.
     */
    @Override
    public CMatrix makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and entries as this matrix.
     */
    @Override
    public CTensor toTensor() {
        return new CTensor(shape, entries.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must be broadcastable to the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same entries as this matrix.
     */
    @Override
    public CTensor toTensor(Shape newShape) {
        ValidateParameters.ensureBroadcastable(shape, newShape);
        return new CTensor(newShape, entries.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape)
     * @see #I(int, int)
     */
    public static CMatrix I(int size) {
        return I(size, size);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int)
     * @see #I(Shape)
     */
    public static CMatrix I(int numRows, int numCols) {
        ValidateParameters.ensureNonNegative(numRows, numCols);
        Complex128[] entries = new Complex128[numRows*numCols];
        Arrays.fill(entries, Complex128.ZERO);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++)
            entries[i*numCols+i] = Complex128.ONE;

        return new CMatrix(new Shape(numRows, numCols), entries);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape Shape of the identity-like matrix.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified shape is not rank 2.
     * @see #I(int)
     * @see #I(int, int)
     */
    public static CMatrix I(Shape shape) {
        ValidateParameters.ensureRank(shape, 2);
        return I(shape.get(0), shape.get(1));
    }


    /**
     * Converts this complex matrix to a real matrix. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real matrix containing the real components of the entries of this matrix.
     */
    public Matrix toReal() {
        double[] re = new double[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            re[i] = ((Complex128) entries[i]).re;

        return new Matrix(shape, re);
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type {@link CMatrix}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CMatrix src2 = (CMatrix) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Gets a row of the matrix formatted as a human-readable string.
     * @param rowIndex Index of the row to get.
     * @param columnsToPrint List of column indices to print.
     * @param maxWidths List of maximum string lengths for each column.
     * @return A human-readable string representation of the specified row.
     */
    private String rowToString(int rowIndex, List<Integer> columnsToPrint, List<Integer> maxWidths) {
        StringBuilder sb = new StringBuilder();

        // Start the row with appropriate bracket.
        sb.append(rowIndex > 0 ? " [" : "[");

        // Loop over the columns to print.
        for (int i = 0; i < columnsToPrint.size(); i++) {
            int colIndex = columnsToPrint.get(i);
            String value;
            int width = PrintOptions.getPadding() + maxWidths.get(i);

            if (colIndex == -1) // Placeholder for truncated columns.
                value = "...";
            else
                value = StringUtils.ValueOfRound(this.get(rowIndex, colIndex), PrintOptions.getPrecision());

            if (PrintOptions.useCentering())
                value = StringUtils.center(value, width);

            sb.append(String.format("%-" + width + "s", value));
        }

        // Close the row.
        sb.append("]");

        return sb.toString();
    }


    /**
     * Generates a human-readable string representing this matrix.
     * @return A human-readable string representing this matrix.
     */
    @Override
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        if (entries.length == 0) {
            result.append("[]"); // No entries in this matrix.
        } else {
            int numRows = this.numRows;
            int numCols = this.numCols;

            int maxRows = PrintOptions.getMaxRows();
            int maxCols = PrintOptions.getMaxColumns();

            int rowStopIndex = Math.min(maxRows - 1, numRows - 1);
            boolean truncatedRows = maxRows < numRows;

            int colStopIndex = Math.min(maxCols - 1, numCols - 1);
            boolean truncatedCols = maxCols < numCols;

            // Build list of column indices to print
            List<Integer> columnsToPrint = new ArrayList<>();
            for (int j = 0; j < colStopIndex; j++)
                columnsToPrint.add(j);

            if (truncatedCols) columnsToPrint.add(-1); // Use -1 to indicate '...'.
            columnsToPrint.add(numCols - 1); // Always include the last column.

            // Compute maximum widths for each column
            List<Integer> maxWidths = new ArrayList<>();
            for (Integer colIndex : columnsToPrint) {
                int maxWidth;
                if (colIndex == -1)
                    maxWidth = 3; // Width for '...'.
                else
                    maxWidth = PrettyPrint.maxStringLength(this.getCol(colIndex).entries, rowStopIndex + 1);

                maxWidths.add(maxWidth);
            }

            // Build the rows up to the stopping index.
            for (int i = 0; i < rowStopIndex; i++) {
                result.append(rowToString(i, columnsToPrint, maxWidths));
                result.append("\n");
            }

            if (truncatedRows) {
                // Print a '...' row to indicate truncated rows.
                int totalWidth = maxWidths.stream().mapToInt(w -> w + PrintOptions.getPadding()).sum();
                String value = "...";

                if (PrintOptions.useCentering())
                    value = StringUtils.center(value, totalWidth);

                result.append(String.format(" [%-" + totalWidth + "s]\n", value));
            }

            // Append the last row.
            result.append(rowToString(numRows - 1, columnsToPrint, maxWidths));
        }

        result.append("]");

        return result.toString();
    }
}
