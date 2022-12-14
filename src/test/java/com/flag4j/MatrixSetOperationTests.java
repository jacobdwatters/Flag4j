package com.flag4j;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatrixSetOperationTests {
    double[][] entriesA, entriesExp;
    Matrix A, exp;

    @Test
    void setValuesDTest() {
        Double[][] values;

        // -------------- Sub-case 1 --------------
        values = new Double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setValuesdTest() {
        double[][] values;

        // -------------- Sub-case 1 --------------
        values = new double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setValuesITest() {
        Integer[][] values;

        // -------------- Sub-case 1 --------------
        values = new Integer[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setValuesintTest() {
        int[][] values;

        // -------------- Sub-case 1 --------------
        values = new int[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new Matrix(values);
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setColumnDTest() {
        Double[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new Double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesExp = new double[][]{{1.345, 0}, {1.5455, 4}, {1.445, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new Double[]{1.345, 1.5455, 1.445};
        col = 1;
        entriesExp = new double[][]{{0, 1.345}, {1, 1.5455}, {1331.14, 1.445}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Double[]{1.345, 1.5455};
        col = -1;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new Double[]{1.345, 1.5455};
        col = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setColumndoubleTest() {
        double[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesExp = new double[][]{{1.345, 0}, {1.5455, 4}, {1.445, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 1;
        entriesExp = new double[][]{{0, 1.345}, {1, 1.5455}, {1331.14, 1.445}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        col = -1;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        col = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setColumnITest() {
        Integer[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new Integer[]{13, 35, -931};
        col = 0;
        entriesExp = new double[][]{{13, 0}, {35, 4}, {-931, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[]{13, 35, -931};
        col = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new Integer[]{13, 35, -931};
        col = 1;
        entriesExp = new double[][]{{0, 13}, {1, 35}, {1331.14, -931}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Integer[]{13, 35};
        col = -1;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new Integer[]{13, 35};
        col = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setColumnintTest() {
        int[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new int[]{13, 35, -931};
        col = 0;
        entriesExp = new double[][]{{13, 0}, {35, 4}, {-931, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[]{13, 35, -931};
        col = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new int[]{13, 35, -931};
        col = 1;
        entriesExp = new double[][]{{0, 13}, {1, 35}, {1331.14, -931}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new int[]{13, 35};
        col = -1;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new int[]{13, 35};
        col = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setRowDTest() {
        Double[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new Double[]{1.345, 1.5455};
        row = 0;
        entriesExp = new double[][]{{1.345, 1.5455}, {1, 4}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Double[]{1.345, 1.5455};
        row = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new Double[]{1.345, 1.5455};
        row = 1;
        entriesExp = new double[][]{{0, 0}, {1.345, 1.5455}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Double[]{1.345, 1.5455};
        row = -1;
        entriesA = new double[][]{{0, 0}, {1, 4}};
        A = new Matrix(entriesA);

        Double[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new Double[]{1.345, 1.5455, 9.45};
        row = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Double[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowdoubleTest() {
        double[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new double[]{1.345, 1.5455};
        row = 0;
        entriesExp = new double[][]{{1.345, 1.5455}, {1, 4}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[]{1.345, 1.5455};
        row = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new double[]{1.345, 1.5455};
        row = 1;
        entriesExp = new double[][]{{0, 0}, {1.345, 1.5455}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        row = -1;
        entriesA = new double[][]{{0, 0}, {1, 4}};
        A = new Matrix(entriesA);

        double[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455, 9.45};
        row = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        double[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowITest() {
        Integer[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new Integer[]{1, 455};
        row = 0;
        entriesExp = new double[][]{{1, 455}, {1, 4}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[]{1, 455};
        row = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new Integer[]{1, 455};
        row = 1;
        entriesExp = new double[][]{{0, 0}, {1, 455}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Integer[]{1, 455};
        row = -1;
        entriesA = new double[][]{{0, 0}, {1, 4}};
        A = new Matrix(entriesA);

        Integer[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new Integer[]{1, 455, 9};
        row = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        Integer[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowintTest() {
        int[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new int[]{1, 455};
        row = 0;
        entriesExp = new double[][]{{1, 455}, {1, 4}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[]{1, 455};
        row = 0;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new int[]{1, 455};
        row = 1;
        entriesExp = new double[][]{{0, 0}, {1, 455}, {1331.14, -1334.5}};
        exp = new Matrix(entriesExp);
        entriesA = new double[][]{{0, 0}, {1, 4}, {1331.14, -1334.5}};
        A = new Matrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new int[]{1, 455};
        row = -1;
        entriesA = new double[][]{{0, 0}, {1, 4}};
        A = new Matrix(entriesA);

        int[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new int[]{1, 455, 9};
        row = 3;
        entriesA = new double[][]{{0, 0, 1}, {1, 4, 2}};
        A = new Matrix(entriesA);

        int[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }

    @Test
    void setSliceMatrixTest() {
        double[][] valueEntries;
        Matrix values;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        Matrix finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopyMatrixTest() {
        double[][] valueEntries;
        Matrix values;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        valueEntries = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        values = new Matrix(valueEntries);
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        Matrix finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }

    @Test
    void setSliceDoubleTest() {
        Double[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        Double[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopyDoubleTest() {
        Double[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new Double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        Double[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSlicedoubleTest() {
        double[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        double[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopydoubleTest() {
        double[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71.33, 34.61, 2.2, 83.1}, {-99.24, -13.4, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        double[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }

    @Test
    void setSliceIntegerTest() {
        Integer[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71, 34, 2.2, 83.1}, {-99, -13, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        Integer[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopyIntegerTest() {
        Integer[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-7, 34, 2.2, 83.1}, {-99, -13, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -7, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        Integer[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }

    @Test
    void setSliceintTest() {
        int[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new int[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-71, 34, 2.2, 83.1}, {-99, -13, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new int[][]{{-71, 34}, {-99, -13}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        int[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopyintTest() {
        int[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new int[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-7, 34, 2.2, 83.1}, {-99, -13, -7.13, 0.00013}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new int[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -7, 34}, {11.346, 124.6, -99, -13}};
        exp = new Matrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new int[][]{{-7, 34}, {-99, -13}};
        entriesA = new double[][]{{-99.234, 132, 2.2, 83.1}, {11.346, 124.6, -7.13, 0.00013}};
        A = new Matrix(entriesA);
        entriesExp = new double[][]{{-99.234, 132, -71.33, 34.61}, {11.346, 124.6, -99.24, -13.4}};
        exp = new Matrix(entriesExp);

        int[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }

}
