package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.io.PrintOptions;
import com.flag4j.util.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Complex Dense Matrix.
 */
public class CMatrix extends TypedMatrix<CNumber[][]> {

    /**
     * Creates an empty complex dense matrix.
     */
    public CMatrix() {
        super(MatrixTypes.C_MATRIX, 0, 0);
        entries = new CNumber[this.m][this.n];
    }


    /**
     * Constructs a square complex dense matrix of a specified size. The entries of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size) {
        super(MatrixTypes.C_MATRIX, size, size);
        this.entries = new CNumber[this.m][this.n];
    }


    /**
     * Creates a square complex dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size, double value) {
        super(MatrixTypes.C_MATRIX, size, size);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = new CNumber();
            }
        }
    }


    /**
     * Creates a square complex dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public CMatrix(int size, CNumber value) {
        super(MatrixTypes.C_MATRIX, size, size);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = value.clone();
            }
        }
    }


    /**
     * Creates a complex dense matrix of a specified shape filled with zeros.
     * @param m The number of rows in the matrix.
     * @param n The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public CMatrix(int m, int n) {
        super(MatrixTypes.C_MATRIX, m, n);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = CNumber.ZERO;
            }
        }
    }


    /**
     * Creates a complex dense matrix with a specified shape and fills the matrix with the specified value.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public CMatrix(int m, int n, double value) {
        super(MatrixTypes.C_MATRIX, m, n);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = new CNumber(value);
            }
        }
    }


    /**
     * Creates a complex dense matrix with a specified shape and fills the matrix with the specified value.
     * @param m Number of rows in the matrix.
     * @param n Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public CMatrix(int m, int n, CNumber value) {
        super(MatrixTypes.C_MATRIX, m, n);
        this.entries = new CNumber[this.m][this.n];

        for(int i=0; i<this.m; i++) {
            for(int j=0; j<this.n; j++) {
                this.entries[i][j] = value.clone();
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(String[][] entries) {
        super(MatrixTypes.C_MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(entries[i][j]);
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(CNumber[][] entries) {
        super(MatrixTypes.C_MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = entries[i][j].clone();
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(double[][] entries) {
        super(MatrixTypes.C_MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(entries[i][j]);
            }
        }
    }


    /**
     * Creates a complex dense matrix whose entries are specified by a double array.
     * @param entries Entries of the real dense matrix.
     */
    public CMatrix(int[][] entries) {
        super(MatrixTypes.C_MATRIX, entries.length, entries[0].length);
        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(entries[i][j]);
            }
        }
    }


    /**
     * Creates a complex dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(Matrix A) {
        super(MatrixTypes.C_MATRIX, A.entries.length, A.entries[0].length);

        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = new CNumber(A.entries[i][j]);
            }
        }
    }


    /**
     * Creates a complex dense matrix which is a copy of a specified matrix.
     * @param A The matrix defining the entries for this matrix.
     */
    public CMatrix(CMatrix A) {
        super(MatrixTypes.C_MATRIX, A.entries.length, A.entries[0].length);

        this.entries = new CNumber[m][n];

        for(int i=0; i<m; i++) {
            for(int j=0; j<n; j++) {
                this.entries[i][j] = A.entries[i][j];
            }
        }
    }


    /**
     * Get the column of this matrix at the specified index.
     * @param j Index of column to get.
     * @return The specified column of this matrix.
     */
    public CNumber[] getCol(int j) {
        CNumber[] column = new CNumber[this.m];

        for(int i=0; i<m; i++) {
            column[i] = this.entries[i][j];
        }

        return column;
    }


    /**
     * Formats matrix contents as a string.
     *
     * @return Matrix as string
     */
    public String toString() {
        String result = "[";

        // Get the current print options.
        int MAX_ROWS = PrintOptions.getMaxRows();
        int MAX_COLUMNS = PrintOptions.getMaxColumns();
        int PADDING = PrintOptions.getPadding();
        int PRECISION = PrintOptions.getPrecision();

        if(!this.isEmpty()) {
            int max=0, colWidth;
            List<Integer> maxList = new ArrayList<>();

            for(int j=0; j<this.n; j++) { // Get the maximum length string representation for each column.
                List<CNumber> contents = Arrays.asList(this.getCol(j));
                Optional<Integer> value = contents.stream().map(CNumber::length).max(Integer::compareTo);

                if(value.isPresent()) {
                    max = value.get();
                }

                maxList.add(max);
            }

            StringBuilder resultBuilder = new StringBuilder("[");
            for(int i = 0; i < m; i++) {
                if(i >= MAX_ROWS && i < m-1) {
                    resultBuilder.append("  ...\n ");
                    i = m-1;
                }

                resultBuilder.append(" [");

                for(int j = 0; j < n; j++) {

                    if(j >= MAX_COLUMNS && j < n-1) {
                        colWidth = 3+PADDING;
                        resultBuilder.append(String.format("%-" + colWidth + "s", StringUtils.center("...", colWidth)));
                        colWidth = maxList.get(n-1)+PADDING;
                        resultBuilder.append(String.format("%-" + (colWidth) + "s", StringUtils.center(entries[i][n - 1].toString(), colWidth)));
                        break;
                    }
                    else {
                        colWidth = maxList.get(j)+PADDING;
                        resultBuilder.append(String.format("%-" + (colWidth) + "s", StringUtils.center(
                                CNumber.round(entries[i][j], PRECISION).toString(), colWidth))
                        );
                    }
                }
                resultBuilder.append("]\n ");
            }
            result = resultBuilder.toString();

            result = result.substring(0, result.length()-2) + " ]";
        }
        else {
            result += "[]]";
        }

        return result;
    }


    public static void main(String[] args)  {
        final String[][] d =
                {{"5", 		"2-2.3i", 	"4.1 + -3.1i"},
                        {"3i", 	"-i", 		"4.1"},
                        {"0+0i",	"0", 		"1"}};
        CMatrix mat = new CMatrix(d);
        System.out.println(mat);
    }
}
