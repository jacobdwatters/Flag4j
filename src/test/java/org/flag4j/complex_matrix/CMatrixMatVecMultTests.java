package org.flag4j.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooCVectorOld;
import org.flag4j.arrays_old.sparse.CooVectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixMatVecMultTests {
    CNumber[][] aEntries;
    CNumber[] expEntries;
    CMatrixOld A;
    CVectorOld exp;


    @Test
    void matMultTestCase() {
        double[][] bEntries;
        VectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{1.666},
                {-0.9345341},
                {0.0}};
        B = new VectorOld(ArrayUtils.flatten(bEntries));
        expEntries = new CNumber[]{new CNumber("163.51005868-15.462680014470001i"),
                new CNumber("1.666+694.4522897100001i"),
                new CNumber("12690.663369999998"),
                new CNumber("-3.377522800415388-8.877571549119406i")};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[][]{{1.666, 11.5},
                {-0.9345341, 88.234},
                {0.0, 2e-05},
                {993.3, 1.23}};
        B = new VectorOld(ArrayUtils.flatten(bEntries));

        VectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultComplexTestCase() {
        CNumber[][] bEntries;
        CVectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i")},
                {new CNumber("-0.0-0.9345341i")},
                {new CNumber("0.0")}};
        B = new CVectorOld(ArrayUtils.flatten(bEntries));
        expEntries = new CNumber[]{
                new CNumber("215.01988001447+65.76525868i"),
                new CNumber("-692.78628971+1.0i"),
                new CNumber("12690.663369999998+7617.445i"),
                new CNumber("16.203765617290802-0.235930146825595i")};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[][]{{new CNumber("1.666+1.0i"), new CNumber("11.5-9.123i")},
                {new CNumber("-0.0-0.9345341i"), new CNumber("88.234")}};
        B = new CVectorOld(ArrayUtils.flatten(bEntries));

        CVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseTestCase() {
        double[] bEntries;
        int[] rowIndices;
        CooVectorOld B;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{-0.9345341};
        rowIndices = new int[]{1};
        B = new CooVectorOld(3, bEntries, rowIndices);
        expEntries = new CNumber[]{
                new CNumber("-42.240941320000005+0.031119985530000005i"),
                new CNumber("0.0+694.4522897100001i"),
                new CNumber("0.0"),
                new CNumber("-8.611416161295983-14.11146491i")};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, A.mult(B));

        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new double[]{-0.9345341};
        rowIndices = new int[]{1};
        B = new CooVectorOld(14, bEntries, rowIndices);

        CooVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }


    @Test
    void matMultSparseComplexTestCase() {
        CNumber[] bEntries;
        int[] rowIndices, colIndices;
        CooCVectorOld B;
        Shape bShape;

        // ---------------------- Sub-case 1 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i")};
        rowIndices = new int[]{1};
        B = new CooCVectorOld(3, bEntries, rowIndices);
        expEntries = new CNumber[]{
                new CNumber("-41.929586320000006+422.65111998553i"),
                new CNumber("6947.985+694.4522897100001i"),
                new CNumber("0.0"),
                new CNumber("-149.79641616129598+72.04562781472501i")};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, A.mult(B));


        // ---------------------- Sub-case 2 ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(123.5, -9.3), new CNumber(45.2, -0.0333), new CNumber(5.4)},
                {new CNumber(1), new CNumber(0, -743.1), new CNumber(-34.5, -93.)},
                {new CNumber(7617.445), new CNumber(0), CNumber.ZERO},
                {new CNumber(Math.PI, Math.PI), new CNumber(9.2146623235, 15.1), new CNumber(-4)}};
        A = new CMatrixOld(aEntries);
        bEntries = new CNumber[]{new CNumber("-0.9345341+9.35i"), new CNumber("11.67-2.0i")};
        rowIndices = new int[]{1687, 2569070};
        B = new CooCVectorOld(3450941, bEntries, rowIndices);

        CooCVectorOld finalB = B;
        assertThrows(LinearAlgebraException.class, ()->A.mult(finalB));
    }

}
