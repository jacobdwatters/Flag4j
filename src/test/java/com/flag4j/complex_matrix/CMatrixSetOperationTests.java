package com.flag4j.complex_matrix;

import com.flag4j.CMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixSetOperationTests {

    CNumber[][] entriesA, entriesExp;
    CMatrix A, exp;

    @Test
    void setValuesCNumberTests() {
        CNumber[][] values;

        // -------------- Sub-case 1 --------------
        values = new CNumber[][]{{new CNumber(23.4, -9.433), new CNumber(-9431, 0.23)},
                {new CNumber(9.23, 55.6), new CNumber(0, -78)},
                {new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};
        exp = new CMatrix(values);
        entriesA = new CNumber[][]{{new CNumber(23.4, -9.433), new CNumber(-9431, 0.23)},
                {new CNumber(9.23, 55.6), new CNumber(0, -78)},
                {new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};
        A = new CMatrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[][]{{new CNumber(23.4, -9.433), new CNumber(-9431, 0.23)},
                {new CNumber(9.23, 55.6), new CNumber(0, -78)},
                {new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};;
        entriesA = new CNumber[][]{{new CNumber(23.4, -9.433), new CNumber(-9431, 0.23), new CNumber(9.23, 55.6)},
                {new CNumber(0, -78), new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};
        A = new CMatrix(entriesA);

        CNumber[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }

    @Test
    void setValuesdTest() {
        double[][] values;

        // -------------- Sub-case 1 --------------
        values = new double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        exp = new CMatrix(values);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[][]{{1.345, 1.5455}, {-0.44, Math.PI}, {13., -9.4}};
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setValuesintTest() {
        int[][] values;

        // -------------- Sub-case 1 --------------
        values = new int[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new CMatrix(values);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setValues(values);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[][]{{1, 55}, {-44, 0}, {13, -9}};
        exp = new CMatrix(values);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[][] finalValues = values;
        assertThrows(IllegalArgumentException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setColumndoubleTest() {
        double[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesExp = new CNumber[][]{{new CNumber(1.345), new CNumber(0)}, {new CNumber(1.5455), new CNumber(4)}, {new CNumber(1.445), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new double[]{1.345, 1.5455, 1.445};
        col = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(1.345)}, {new CNumber(1), new CNumber(1.5455)}, {new CNumber(1331.14), new CNumber(1.445)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        col = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        col = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[] finalValues2 = values;
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
        entriesExp = new CNumber[][]{{new CNumber(13), new CNumber(0)}, {new CNumber(35), new CNumber(4)}, {new CNumber(-931), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[]{13, 35, -931};
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new int[]{13, 35, -931};
        col = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(13)}, {new CNumber(1), new CNumber(35)}, {new CNumber(1331.14), new CNumber(-931)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new int[]{13, 35};
        col = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new int[]{13, 35};
        col = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }



    @Test
    void setColumnCNumberTest() {
        CNumber[] values;
        int col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        col = 0;
        entriesExp = new CNumber[][]
                {{new CNumber(2.345, 5.15), new CNumber(0)},
                {new CNumber(-445, 0.32), new CNumber(4)},
                {new CNumber(94.1), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(2.345, 5.15), new CNumber(0), new CNumber(1)},
                {new CNumber(-445, 0.32), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        col = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(2.345, 5.15)},
                {new CNumber(1), new CNumber(-445, 0.32)},
                {new CNumber(1331.14), new CNumber(94.1)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32)};
        col = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32)};
        col = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues2 = values;
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
        entriesExp = new CNumber[][]{
                {new CNumber(13), new CNumber(0, 45.6)},
                {new CNumber(35), new CNumber(4, -13.1)},
                {new CNumber(-931), new CNumber(-1334.5, 0.0043)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0, 4), new CNumber(0, 45.6)},
                {new CNumber(1, 66.712), new CNumber(4, -13.1)},
                {new CNumber(1331.14, -92.23), new CNumber(-1334.5, 0.0043)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[]{13, 35, -931};
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(5.212, -67.1), new CNumber(5.234, 1), new CNumber(66, 1)},
                {new CNumber(-0.432, -82.24), new CNumber(6.2, 14678.324), new CNumber(0, 34)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues = values;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValues, finalCol));

        // -------------- Sub-case 3 --------------
        values = new Integer[]{13, 35, -931};
        col = 1;
        entriesExp = new CNumber[][]{
                {new CNumber(0, 4), new CNumber(13)},
                {new CNumber(1, 66.712), new CNumber(35)},
                {new CNumber(1331.14, -92.23), new CNumber(-931)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0, 4), new CNumber(0, 45.6)},
                {new CNumber(1, 66.712), new CNumber(4, -13.1)},
                {new CNumber(1331.14, -92.23), new CNumber(-1334.5, 0.0043)}};
        A = new CMatrix(entriesA);
        A.setCol(values, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Integer[]{13, 35};
        col = -1;
        entriesA = new CNumber[][]{
                {new CNumber(5.212, -67.1), new CNumber(5.234, 1), new CNumber(66, 1)},
                {new CNumber(-0.432, -82.24), new CNumber(6.2, 14678.324), new CNumber(0, 34)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues1 = values;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new Integer[]{13, 35};
        col = 3;
        entriesA = new CNumber[][]{
                {new CNumber(5.212, -67.1), new CNumber(5.234, 1), new CNumber(66, 1)},
                {new CNumber(-0.432, -82.24), new CNumber(6.2, 14678.324), new CNumber(0, 34)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues2 = values;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValues2, finalCol2));
    }


    @Test
    void setRowdoubleTest() {
        double[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new double[]{1.345, 1.5455};
        row = 0;
        entriesExp = new CNumber[][]{{new CNumber(1.345), new CNumber(1.5455)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[]{1.345, 1.5455};
        row = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new double[]{1.345, 1.5455};
        row = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1.345), new CNumber(1.5455)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455};
        row = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}};
        A = new CMatrix(entriesA);

        double[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new double[]{1.345, 1.5455, 9.45};
        row = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        double[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowCNumberTest() {
        CNumber[] values;
        int row;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        row = 0;
        entriesExp = new CNumber[][]{
                {new CNumber(34, -55.6), new CNumber(0.44, -0.23)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        row = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        row = 1;
        entriesExp = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(34, -55.6), new CNumber(0.44, -0.23)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        row = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23), new CNumber(9.234, -0.2334)};
        row = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues2 = values;
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
        entriesExp = new CNumber[][]{{new CNumber(1), new CNumber(455)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[]{1, 455};
        row = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new int[]{1, 455};
        row = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(455)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new int[]{1, 455};
        row = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}};
        A = new CMatrix(entriesA);

        int[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new int[]{1, 455, 9};
        row = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        int[] finalValues2 = values;
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
        entriesExp = new CNumber[][]{{new CNumber(1), new CNumber(455)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[]{1, 455};
        row = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new Integer[]{1, 455};
        row = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(455)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(values, row);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new Integer[]{1, 455};
        row = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues1 = values;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new Integer[]{1, 455, 9};
        row = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        Integer[] finalValues2 = values;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setSliceMatrixTest() {
        CNumber[][] valueEntries;
        CMatrix values;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99.24), new CNumber(-13.4), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        CMatrix finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }

    @Test
    void setSliceCopyMatrixTest() {
        CNumber[][] valueEntries;
        CMatrix values;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99.24), new CNumber(-13.4), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        valueEntries = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(-99.24), new CNumber(-13.4)}};
        values = new CMatrix(valueEntries);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        CMatrix finalValues = values;
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
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99.24), new CNumber(-13.4), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

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
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71.33), new CNumber(34.61), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99.24), new CNumber(-13.4), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new double[][]{{-71.33, 34.61}, {-99.24, -13.4}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        double[][] finalValues = values;
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
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71), new CNumber(34), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99), new CNumber(-13), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new int[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new int[][]{{-71, 34}, {-99, -13}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

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
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-7), new CNumber(34), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99), new CNumber(-13), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new int[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-7), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new int[][]{{-7, 34}, {-99, -13}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        int[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceITest() {
        Integer[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-71), new CNumber(34), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99), new CNumber(-13), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new Integer[][]{{-71, 34}, {-99, -13}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        Integer[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyITest() {
        Integer[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-7), new CNumber(34), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(-99), new CNumber(-13), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-7), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new Integer[][]{{-7, 34}, {-99, -13}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        Integer[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceCNumberTest() {
        CNumber[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(0, 442.4), new CNumber(24.88), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0, 442.4), new CNumber(24.88)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        CNumber[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyCNumberTest() {
        CNumber[][] values;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(0, 442.4), new CNumber(24.88), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 2 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0, 442.4), new CNumber(24.88)}};
        exp = new CMatrix(entriesExp);

        assertEquals(exp, A.setSliceCopy(values, row, col));

        // -------------- Sub-case 3 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);

        CNumber[][] finalValues = values;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }
}
