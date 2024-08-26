package org.flag4j.complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixRowColSumTests {

    int[] indices;
    CNumber[][] aEntries;
    CNumber[][] expEntries;
    CNumber[] expVecEntries;
    CMatrixOld A;
    CMatrixOld exp;
    CVectorOld expVec;


    @Test
    void sumRowsTestCase() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        expVecEntries = new CNumber[]{new CNumber("7976.594999999999-354.3i"), new CNumber("54.4146623235-728.0333i"), new CNumber(
                "-33.1-93.0i")};
        expVec = new CVectorOld(expVecEntries);

        assertEquals(expVec, A.sumRows());
    }


    @Test
    void addToEachRowRealTestCase() {
        double[] bEntries;
        VectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{1.445, -775.14, 9.4};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[1]),
                        new CNumber(5.4).add(bEntries[2])},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[2])},
                {new CNumber(7617.445).add(bEntries[0]), new CNumber(0).add(bEntries[1]), CNumber.ZERO.add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[0]), new CNumber(9.2146623235, 15.1).add(bEntries[1]), new CNumber(-4).add(bEntries[2])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{23.456};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, () -> A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowComplexTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[1]), new CNumber(5.4).add(bEntries[2])},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[2])},
                {new CNumber(7617.445).add(bEntries[0]), new CNumber(0).add(bEntries[1]), CNumber.ZERO.add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[0]), new CNumber(9.2146623235, 15.1).add(bEntries[1]), new CNumber(-4).add(bEntries[2])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34), new CNumber(84.35, -6767)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowComplexSparseTestCase() {
        CNumber[] bEntries;
        CooCVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(32.56, -8.4)};
        indices = new int[]{1};
        b = new CooCVectorOld(3, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0).add(bEntries[0]), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1).add(bEntries[0]), new CNumber(-4)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(32.56, -8.4)};
        indices = new int[]{1};
        b = new CooCVectorOld(234, bEntries, indices);

        CooCVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void addToEachRowRealSparseTestCase() {
        double[] bEntries;
        CooVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{3.46567};
        indices = new int[]{1};
        b = new CooVectorOld(3, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0).add(bEntries[0]), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1).add(bEntries[0]), new CNumber(-4)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachRow(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{-9899234.2};
        indices = new int[]{1};
        b = new CooVectorOld(234, bEntries, indices);

        CooVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachRow(finalB));
    }


    @Test
    void sumColsTestCase() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        expVecEntries = new CNumber[]{
                new CNumber("174.1-9.333300000000001i"),
                new CNumber("-33.5-836.1i"),
                new CNumber("7617.445"),
                new CNumber("239.8646623235-329.9i")};
        expVec = new CVectorOld(expVecEntries);

        assertEquals(expVec, A.sumCols());
    }


    @Test
    void addToEachColRealTestCase() {
        double[] bEntries;
        VectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{234.66, -8.54, 9.45, 16};
        b = new VectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4).add(bEntries[0])},
                {new CNumber(1).add(bEntries[1]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[1])},
                {new CNumber(7617.445).add(bEntries[2]), new CNumber(0).add(bEntries[2]), CNumber.ZERO.add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[3]), new CNumber(9.2146623235, 15.1).add(bEntries[3]), new CNumber(-4).add(bEntries[3])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{234.66, -8.54, 9.45};
        b = new VectorOld(bEntries);

        VectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColComplexTestCase() {
        CNumber[] bEntries;
        CVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34), new CNumber(84.35, -6767)};
        b = new CVectorOld(bEntries);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3).add(bEntries[0]), new CNumber(45.2, -0.0333).add(bEntries[0]), new CNumber(5.4).add(bEntries[0])},
                {new CNumber(1).add(bEntries[1]), new CNumber(0, -743.1).add(bEntries[1]), new CNumber(-34.5, -93.).add(bEntries[1])},
                {new CNumber(7617.445).add(bEntries[2]), new CNumber(0).add(bEntries[2]), CNumber.ZERO.add(bEntries[2])},
                {new CNumber(234.65, -345.).add(bEntries[3]), new CNumber(9.2146623235, 15.1).add(bEntries[3]), new CNumber(-4).add(bEntries[3])}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(234.6, 8.4), new CNumber(0.345, -9), new CNumber(23.56, -7.34)};
        b = new CVectorOld(bEntries);

        CVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColRealSparseTestCase() {
        double[] bEntries;
        CooVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{234.66};
        indices = new int[]{2};
        b = new CooVectorOld(4, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445).add(234.66), new CNumber(0).add(234.66), CNumber.ZERO.add(234.66)},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{234.66};
        indices = new int[]{2};
        b = new CooVectorOld(234, bEntries, indices);

        CooVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }


    @Test
    void addToEachColComplexSparseTestCase() {
        CNumber[] bEntries;
        CooCVectorOld b;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(3.678, -8.4322)};
        indices = new int[]{1};
        b = new CooCVectorOld(4, bEntries, indices);
        expEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1).add(bEntries[0]), new CNumber(0, -743.1).add(bEntries[0]), new CNumber(-34.5, -93.).add(bEntries[0])},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        exp = new CMatrixOld(expEntries);

        assertEquals(exp, A.addToEachCol(b));

        // ------------------------ Sub-case 2 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(234.65, -345.), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber(3.678, -8.4322)};
        indices = new int[]{1};
        b = new CooCVectorOld(234, bEntries, indices);

        CooCVectorOld finalB = b;
        assertThrows(IllegalArgumentException.class, ()->A.addToEachCol(finalB));
    }
}
