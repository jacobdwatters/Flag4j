package org.flag4j.complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.core.Shape;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixSetOperationTests {

    CNumber[][] entriesA, entriesExp;
    CMatrix A, exp;
    int sparseSize;
    int[] sparseIndices;
    int[] rowIndices, colIndices;
    Shape sparseShape;

    @Test
    void setValuesCNumberTestCase() {
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
                {new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};
        entriesA = new CNumber[][]{{new CNumber(23.4, -9.433), new CNumber(-9431, 0.23), new CNumber(9.23, 55.6)},
                {new CNumber(0, -78), new CNumber(5.1114, -5821.23), new CNumber(754.1, -823.1)}};
        A = new CMatrix(entriesA);

        CNumber[][] finalValues = values;
        assertThrows(LinearAlgebraException.class, () -> A.setValues(finalValues));
    }

    @Test
    void setValuesdTestCase() {
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
        assertThrows(LinearAlgebraException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setValuesintTestCase() {
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
        assertThrows(LinearAlgebraException.class, () -> A.setValues(finalValues));
    }


    @Test
    void setColumndoubleTestCase() {
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
    void setColumnintTestCase() {
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
    void setColumnCNumberTestCase() {
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
    void setColumnITestCase() {
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
    void setColumnCVectorTestCase() {
        CNumber[] values;
        CVector valuesVec;
        int col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        valuesVec = new CVector(values);
        col = 0;
        entriesExp = new CNumber[][]
                {{new CNumber(2.345, 5.15), new CNumber(0)},
                        {new CNumber(-445, 0.32), new CNumber(4)},
                        {new CNumber(94.1), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        valuesVec = new CVector(values);
        col = 0;
        entriesA = new CNumber[][]{{new CNumber(2.345, 5.15), new CNumber(0), new CNumber(1)},
                {new CNumber(-445, 0.32), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec = valuesVec;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, () -> A.setCol(finalValuesVec, finalCol));

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32), new CNumber(94.1)};
        valuesVec = new CVector(values);
        col = 1;
        entriesExp = new CNumber[][]{{new CNumber(0), new CNumber(2.345, 5.15)},
                {new CNumber(1), new CNumber(-445, 0.32)},
                {new CNumber(1331.14), new CNumber(94.1)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32)};
        valuesVec = new CVector(values);
        col = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec1 = valuesVec;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValuesVec1, finalCol1));

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(2.345, 5.15), new CNumber(-445, 0.32)};
        valuesVec = new CVector(values);
        col = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CVector finalValuesVec2 = valuesVec;
        int finalCol2 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setCol(finalValuesVec2, finalCol2));
    }


    @Test
    void setColumnSparseCVectorTestCase() {
        CNumber[] values;
        CooCVector valuesVec;
        int col;

        // ----------------------- Sub-case 1 -----------------------
        values = new CNumber[]{new CNumber(2.445, -0.91354), new CNumber(0, 6.2132)};
        sparseSize = 3;
        sparseIndices = new int[]{0, 2};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        col = 0;
        entriesExp = new CNumber[][]{
                {new CNumber(2.445, -0.91354), new CNumber(0)},
                {CNumber.ZERO, new CNumber(4)},
                {new CNumber(0, 6.2132), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // ----------------------- Sub-case 2 -----------------------
        values = new CNumber[]{new CNumber(2.445, -0.91354)};
        sparseSize = 3;
        sparseIndices = new int[]{1};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        col = 1;
        entriesExp = new CNumber[][]{
                {new CNumber(0), CNumber.ZERO},
                {new CNumber(1), new CNumber(2.445, -0.91354)},
                {new CNumber(1331.14), CNumber.ZERO}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setCol(valuesVec, col);

        assertEquals(exp, A);

        // ----------------------- Sub-case 3 -----------------------
        values = new CNumber[]{new CNumber(2.445, -0.91354)};
        sparseSize = 4;
        sparseIndices = new int[]{1};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);

        CooCVector finalValuesVec = valuesVec;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, ()->A.setCol(finalValuesVec, finalCol));


        // ----------------------- Sub-case 4 -----------------------
        values = new CNumber[]{new CNumber(2.445, -0.91354)};
        sparseSize = 3;
        sparseIndices = new int[]{1};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        col = 13;
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);

        CooCVector finalValuesVec1 = valuesVec;
        int finalCol1 = col;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setCol(finalValuesVec1, finalCol1));
    }


    @Test
    void setRowdoubleTestCase() {
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
    void setRowCNumberTestCase() {
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
    void setRowintTestCase() {
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
    void setRowITestCase() {
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
    void setRowCVectorTestCase() {
        CNumber[] values;
        CVector valuesVec;
        int row;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = 0;
        entriesExp = new CNumber[][]{
                {new CNumber(34, -55.6), new CNumber(0.44, -0.23)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}, {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(valuesVec, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        valuesVec = new CVector(values);
        row = 0;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CNumber[] finalValues = values;
        int finalRow = row;
        assertThrows(IllegalArgumentException.class, () -> A.setRow(finalValues, finalRow));

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23)};
        valuesVec = new CVector(values);
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
        valuesVec = new CVector(values);
        row = -1;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0)}, {new CNumber(1), new CNumber(4)}};
        A = new CMatrix(entriesA);

        CVector finalValues1 = valuesVec;
        int finalRow1 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues1, finalRow1));

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(34, -55.6), new CNumber(0.44, -0.23), new CNumber(9.234, -0.2334)};
        valuesVec = new CVector(values);
        row = 3;
        entriesA = new CNumber[][]{{new CNumber(0), new CNumber(0), new CNumber(1)}, {new CNumber(1), new CNumber(4), new CNumber(2)}};
        A = new CMatrix(entriesA);

        CVector finalValues2 = valuesVec;
        int finalRow2 = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> A.setRow(finalValues2, finalRow2));
    }


    @Test
    void setRowSparseCVectorTestCase() {
        CNumber[] values;
        CooCVector valuesVec;
        int row;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(34, -55.6)};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        row = 0;
        entriesExp = new CNumber[][]{
                {new CNumber(34, -55.6), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(valuesVec, row);

        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(34, -55.6)};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        row = 2;
        entriesExp = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(34, -55.6), CNumber.ZERO}};
        exp = new CMatrix(entriesExp);
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);
        A.setRow(valuesVec, row);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(34, -55.6)};
        sparseSize = 2;
        sparseIndices = new int[]{0};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        row = 3;
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);

        CooCVector finalValuesVec = valuesVec;
        int finalRow = row;
        assertThrows(ArrayIndexOutOfBoundsException.class, ()->A.setRow(finalValuesVec, finalRow));

        // -------------- Sub-case 4 --------------
        values = new CNumber[]{new CNumber(34, -55.6)};
        sparseSize = 13;
        sparseIndices = new int[]{0};
        valuesVec = new CooCVector(sparseSize, values, sparseIndices);
        row = 1;
        entriesA = new CNumber[][]{
                {new CNumber(0), new CNumber(0)},
                {new CNumber(1), new CNumber(4)},
                {new CNumber(1331.14), new CNumber(-1334.5)}};
        A = new CMatrix(entriesA);

        CooCVector finalValuesVec2 = valuesVec;
        int finalRow2 = row;
        assertThrows(IllegalArgumentException.class, ()->A.setRow(finalValuesVec2, finalRow2));
    }


    @Test
    void setSliceMatrixTestCase() {
        double[][] valueEntries;
        Matrix values;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new double[][]{{1, -9.4}, {0.0024, 51.5}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(1), new CNumber(-9.4), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(0.0024), new CNumber(51.5), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        valueEntries = new double[][]{{1.234, -9.4}, {0.0024, 51.5}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(1.234), new CNumber(-9.4)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0.0024), new CNumber(51.5)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        valueEntries = new double[][]{{1.234, -9.4, 24.5, 1}, {0.0024, 51.5, -0.924, 51.6}};
        values = new Matrix(valueEntries);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        Matrix finalValues = values;
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyMatrixTestCase() {
        double[][] valueEntries;
        Matrix values;
        CMatrix B;
        int row, col;

        // -------------- Sub-case 1 --------------
        valueEntries = new double[][]{{1, -9.4}, {0.0024, 51.5}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(1), new CNumber(-9.4), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(0.0024), new CNumber(51.5), new CNumber(-7.13), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(values, row, col);
        assertEquals(exp, B);

        // -------------- Sub-case 2 --------------
        valueEntries = new double[][]{{1.234, -9.4}, {0.0024, 51.5}};
        values = new Matrix(valueEntries);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(1.234), new CNumber(-9.4)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0.0024), new CNumber(51.5)}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(values, row, col);
        assertEquals(exp, B);

        // -------------- Sub-case 3 --------------
        valueEntries = new double[][]{{1.234, -9.4, 24.5, 1}, {0.0024, 51.5, -0.924, 51.6}};
        values = new Matrix(valueEntries);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71.33), new CNumber(34.61)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99.24), new CNumber(-13.4)}};
        exp = new CMatrix(entriesExp);

        Matrix finalValues = values;
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceSparseMatrixTestCase() {
        double[] valueEntries;
        CooMatrix values;
        int row, col;

        // ----------------------- Sub-case 1 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 3);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{2, 0, 2};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(1.3), new CNumber(83.1)},
                {new CNumber(5.626), CNumber.ZERO, new CNumber(-3.0001), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // ----------------------- Sub-case 2 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), CNumber.ZERO, new CNumber(1.3), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber( 5.626), new CNumber(-3.0001), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(values, row, col);
        assertEquals(exp, A);

        // ----------------------- Sub-case 3 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(21, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);

        CooMatrix finalValues = values;
        int finalRow = row;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, finalRow, finalCol));

        // ----------------------- Sub-case 4 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 3;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);

        CooMatrix finalValues1 = values;
        int finalRow1 = row;
        int finalCol1 = col;
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues1, finalRow1, finalCol1));
    }


    @Test
    void setSliceCopySparseMatrixTestCase() {
        double[] valueEntries;
        CooMatrix values;
        CMatrix B;
        int row, col;

        // ----------------------- Sub-case 1 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 3);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{2, 0, 2};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(1.3), new CNumber(83.1)},
                {new CNumber(5.626), CNumber.ZERO, new CNumber(-3.0001), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(values, row, col);
        assertEquals(exp, B);

        // ----------------------- Sub-case 2 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), CNumber.ZERO, new CNumber(1.3), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber( 5.626), new CNumber(-3.0001), new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(values, row, col);
        assertEquals(exp, B);

        // ----------------------- Sub-case 3 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(21, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 0;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);

        CooMatrix finalValues = values;
        int finalRow = row;
        int finalCol = col;
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, finalRow, finalCol));

        // ----------------------- Sub-case 4 -----------------------
        valueEntries = new double[]{1.3, 5.626, -3.0001};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1, 1};
        colIndices = new int[]{1, 0, 1};
        values = new CooMatrix(sparseShape, valueEntries, rowIndices, colIndices);
        row = 3;
        col = 1;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);

        CooMatrix finalValues1 = values;
        int finalRow1 = row;
        int finalCol1 = col;
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues1, finalRow1, finalCol1));
    }


    @Test
    void setSliceCopyCMatrixTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSlicedoubleTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopydoubleTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceintTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyintTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceITestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyITestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceCNumberTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopyCNumberTestCase() {
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
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }


    @Test
    void setSliceCMatrixTestCase() {
        CNumber[][] values;
        CMatrix mat;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        mat = new CMatrix(values);
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

        A.setSlice(mat, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        mat = new CMatrix(values);
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

        A.setSlice(mat, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new CNumber[][]{
                {new CNumber(0.234, -84.12), new CNumber(33, 441.435)},
                {new CNumber(0, 442.4), new CNumber(24.88)}};
        mat = new CMatrix(values);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        CMatrix finalValues = mat;
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceSparseCMatrixTestCase() {
        CNumber[] values;
        CooCMatrix mat;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(2, 3);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(234.5, -99.234), new CNumber(83.1)},
                {CNumber.ZERO, new CNumber(0, -88.245), CNumber.ZERO, new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        A.setSlice(mat, row, col);
        assertEquals(exp, A);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 0};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), CNumber.ZERO, new CNumber(234.5, -99.234)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0, -88.245), CNumber.ZERO}};
        exp = new CMatrix(entriesExp);

        A.setSlice(mat, row, col);

        assertEquals(exp, A);

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(15, 60);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 0};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        CooCMatrix finalValues = mat;
        assertThrows(IllegalArgumentException.class, ()->A.setSlice(finalValues, 1, 2));
    }


    @Test
    void setSliceCopySparseCMatrixTestCase() {
        CNumber[] values;
        CooCMatrix mat;
        CMatrix B;
        int row, col;

        // -------------- Sub-case 1 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(2, 3);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{2, 1};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        row = 0;
        col = 0;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {CNumber.ZERO, CNumber.ZERO, new CNumber(234.5, -99.234), new CNumber(83.1)},
                {CNumber.ZERO, new CNumber(0, -88.245), CNumber.ZERO, new CNumber(0.00013)}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(mat, row, col);
        assertEquals(exp, B);

        // -------------- Sub-case 2 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(2, 2);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 0};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        row = 0;
        col = 2;
        entriesA = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{
                {new CNumber(-99.234), new CNumber(132), CNumber.ZERO, new CNumber(234.5, -99.234)},
                {new CNumber(11.346), new CNumber(124.6), new CNumber(0, -88.245), CNumber.ZERO}};
        exp = new CMatrix(entriesExp);

        B = A.setSliceCopy(mat, row, col);
        assertEquals(exp, B);

        // -------------- Sub-case 3 --------------
        values = new CNumber[]{new CNumber(234.5, -99.234), new CNumber(0, -88.245)};
        sparseShape = new Shape(15, 60);
        rowIndices = new int[]{0, 1};
        colIndices = new int[]{1, 0};
        mat = new CooCMatrix(sparseShape, values, rowIndices, colIndices);
        entriesA = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(2.2), new CNumber(83.1)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-7.13), new CNumber(0.00013)}};
        A = new CMatrix(entriesA);
        entriesExp = new CNumber[][]{{new CNumber(-99.234), new CNumber(132), new CNumber(-71), new CNumber(34)}, {new CNumber(11.346), new CNumber(124.6), new CNumber(-99), new CNumber(-13)}};
        exp = new CMatrix(entriesExp);

        CooCMatrix finalValues = mat;
        assertThrows(IllegalArgumentException.class, ()->A.setSliceCopy(finalValues, 1, 2));
    }
}
