package com.flag4j.complex_matrix;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CMatrixRowColSumTests {

    int[] indices;
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;


    @Test
    void sumRowsTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{{new CNumber("7976.594999999999-354.3i"), new CNumber("54.4146623235-728.0333i"), new CNumber("-33.1-93.0i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sumRows());
    }


    @Test
    void addToEachRowRealTest() {
        double[] bEntries;
        Vector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{1.445, -775.14, 9.4};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[1]), new CNumber(5.4).add(bEntries[2])},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[2])},
                {new CNumber(7617.445).add(bEntries[0]), new CNumber(0).add(bEntries[1]), new CNumber().add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[0]), new CNumber(9.2146623235, 15.1).add(bEntries[1]), new CNumber(-4).add(bEntries[2])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{23.456};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, () -> A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowComplexTest() {
        CNumber[] bEntries;
        CVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[1]), new CNumber(5.4).add(bEntries[2])},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[2])},
                {new CNumber(7617.445).add(bEntries[0]), new CNumber(0).add(bEntries[1]), new CNumber().add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[0]), new CNumber(9.2146623235, 15.1).add(bEntries[1]), new CNumber(-4).add(bEntries[2])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34), new CNumber(84.35, -6767)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowComplexSparseTest() {
        CNumber[] bEntries;
        SparseCVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(32.56, -8.4)};
        indices = new int[]{1};
        b = new SparseCVector(3, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0).add(bEntries[0]), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1).add(bEntries[0]), new CNumber(-4)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(32.56, -8.4)};
        indices = new int[]{1};
        b = new SparseCVector(234, bEntries, indices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowRealSparseTest() {
        double[] bEntries;
        SparseVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{3.46567};
        indices = new int[]{1};
        b = new SparseVector(3, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0).add(bEntries[0]), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1).add(bEntries[0]), new CNumber(-4)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{-9899234.2};
        indices = new int[]{1};
        b = new SparseVector(234, bEntries, indices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void sumColsTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber("174.1-9.333300000000001i")},
                {new CNumber("-33.5-836.1i")},
                {new CNumber("7617.445")},
                {new CNumber("239.8646623235-329.9i")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sumCols());
    }


    @Test
    void addToEachColRealTest() {
        double[] bEntries;
        Vector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{234.66, -8.54, 9.45, 16};
        b = new Vector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4).add(bEntries[0])},
                {new CNumber(1).add(bEntries[1]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[1])},
                {new CNumber(7617.445).add(bEntries[2]), new CNumber(0).add(bEntries[2]), new CNumber().add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[3]), new CNumber(9.2146623235, 15.1).add(bEntries[3]), new CNumber(-4).add(bEntries[3])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{234.66, -8.54, 9.45};
        b = new Vector(bEntries);

        Vector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColComplexTest() {
        CNumber[] bEntries;
        CVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34), new CNumber(84.35, -6767)};
        b = new CVector(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4).add(bEntries[0])},
                {new CNumber(1).add(bEntries[1]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[1])},
                {new CNumber(7617.445).add(bEntries[2]), new CNumber(0).add(bEntries[2]), new CNumber().add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[3]), new CNumber(9.2146623235, 15.1).add(bEntries[3]), new CNumber(-4).add(bEntries[3])}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34)};
        b = new CVector(bEntries);

        CVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColRealSparseTest() {
        double[] bEntries;
        SparseVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{234.66};
        indices = new int[]{2};
        b = new SparseVector(4, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445).add(234.66), new CNumber(0).add(234.66), new CNumber().add(234.66)},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new double[]{234.66};
        indices = new int[]{2};
        b = new SparseVector(234, bEntries, indices);

        SparseVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColComplexSparseTest() {
        CNumber[] bEntries;
        SparseCVector b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(3.678, -8.4322)};
        indices = new int[]{1};
        b = new SparseCVector(4, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.).add(bEntries[0])},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), new CNumber()},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrix(aEntries);
        bEntries = new CNumber[]{new CNumber(3.678, -8.4322)};
        indices = new int[]{1};
        b = new SparseCVector(234, bEntries, indices);

        SparseCVector finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }
}
