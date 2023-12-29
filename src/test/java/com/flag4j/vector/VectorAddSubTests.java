package com.flag4j.vector;

import com.flag4j.CVector;
import com.flag4j.CooCVector;
import com.flag4j.CooVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VectorAddSubTests {

    int[] indices;
    int size;

    double[] aEntries;
    Vector A;

    @Test
    void realDenseAddTestCase() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]+bEntries[0], aEntries[1]+bEntries[1], aEntries[2]+bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void ComplexDenseAddTestCase() {
        CNumber[] bEntries, expEntries;
        CVector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34.56, -0.9345), new CNumber(4.666, 1), new CNumber(0, 8.4)};
        B = new CVector(bEntries);
        expEntries = new CNumber[]{bEntries[0].add(aEntries[0]), bEntries[1].add(aEntries[1]), bEntries[2].add(aEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34.56, -0.9345), new CNumber(4.666, 1)};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void realSparseAddTestCase() {
        double[] bEntries, expEntries;
        CooVector B;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVector(size, bEntries, indices);
        expEntries = new double[]{1.34+34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void complexSparseAddTestCase() {
        CNumber[] bEntries, expEntries;
        CooCVector B;
        CVector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(345.66, -1.44559)};
        indices = new int[]{0};
        size = 3;
        B = new CooCVector(size, bEntries, indices);
        expEntries = new CNumber[]{bEntries[0].add(1.34), new CNumber(6.266), new CNumber(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(345.66, -1.44559)};
        indices = new int[]{0};
        size = 201;
        B = new CooCVector(size, bEntries, indices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.add(finalB));
    }


    @Test
    void doubleAddTestCase() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34+B, 6.266+B, -90.45+B};
        exp = new Vector(expEntries);

        assertEquals(exp, A.add(B));
    }


    @Test
    void cNumberAddTestCase() {
        CNumber[] expEntries;
        CNumber B = new CNumber(5.666, 0.975);
        CVector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new CNumber[]{B.add(1.34), B.add(6.266), B.add(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.add(B));
    }


    @Test
    void realDenseSubTestCase() {
        double[] bEntries, expEntries;
        Vector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7};
        B = new Vector(bEntries);
        expEntries = new double[]{aEntries[0]-bEntries[0], aEntries[1]-bEntries[1], aEntries[2]-bEntries[2]};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677, -8.51, 56.7, 1.34};
        B = new Vector(bEntries);

        Vector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void ComplexDenseSubTestCase() {
        CNumber[] bEntries, expEntries;
        CVector B, exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34.56, -0.9345), new CNumber(4.666, 1), new CNumber(0, 8.4)};
        B = new CVector(bEntries);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).sub(bEntries[0]), new CNumber(aEntries[1]).sub(bEntries[1]), new CNumber(aEntries[2]).sub(bEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(34.56, -0.9345), new CNumber(4.666, 1)};
        B = new CVector(bEntries);

        CVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void realSparseSubTestCase() {
        double[] bEntries, expEntries;
        CooVector B;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 3;
        B = new CooVector(size, bEntries, indices);
        expEntries = new double[]{1.34-34.677, 6.266, -90.45};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new double[]{34.677};
        indices = new int[]{0};
        size = 201;
        B = new CooVector(size, bEntries, indices);

        CooVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void complexSparseSubTestCase() {
        CNumber[] bEntries, expEntries;
        CooCVector B;
        CVector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(345.66, -1.44559)};
        indices = new int[]{0};
        size = 3;
        B = new CooCVector(size, bEntries, indices);
        expEntries = new CNumber[]{new CNumber(1.34).sub(bEntries[0]), new CNumber(6.266), new CNumber(-90.45)};
        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));

        // -------------------- Sub-case 2 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        bEntries = new CNumber[]{new CNumber(345.66, -1.44559)};
        indices = new int[]{0};
        size = 201;
        B = new CooCVector(size, bEntries, indices);

        CooCVector finalB = B;
        assertThrows(LinearAlgebraException.class, () -> A.sub(finalB));
    }


    @Test
    void doubleSubTestCase() {
        double[] expEntries;
        double B = 1.5;
        Vector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new double[]{1.34-B, 6.266-B, -90.45-B};
        exp = new Vector(expEntries);

        assertEquals(exp, A.sub(B));
    }


    @Test
    void cNumberSubTestCase() {
        CNumber[] expEntries;
        CNumber B = new CNumber(5.666, 0.975);
        CVector exp;

        // -------------------- Sub-case 1 --------------------
        aEntries = new double[]{1.34, 6.266, -90.45};
        A = new Vector(aEntries);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).sub(B), new CNumber(aEntries[1]).sub(B), new CNumber(aEntries[2]).sub(B)};

        exp = new CVector(expEntries);

        assertEquals(exp, A.sub(B));
    }
}
