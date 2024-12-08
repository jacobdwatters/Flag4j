package org.flag4j.io;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MatrixMarketReadTests {

    @Test
    void loadRealDenseTestCase() throws IOException {
        String fileName;
        Matrix exp;
        Matrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/data/matrix_market/ReDe3x3.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe3x3);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/data/matrix_market/ReDe2x4.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe2x4);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 3 ----------------
        fileName = "./src/test/data/matrix_market/ReDe3x2.mtx";
        exp = new Matrix(MatrixMarketMatrices.ReDe3x2);
        act = (Matrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 4 ----------------
        fileName = "./src/test/data/matrix_market/ReDe49x15.mtx";

        exp = new Matrix(MatrixMarketMatrices.ReDe49x15);
        act = (Matrix) mmParser.read(fileName);
        assertEquals(exp, act);
    }


    @Test
    void loadComplexDenseTestCase() throws IOException {
        String fileName;
        CMatrix exp;
        CMatrix act;
        MatrixMarketReader mmParser = new MatrixMarketReader();

        // ---------------- sub-case 1 ----------------
        fileName = "./src/test/data/matrix_market/CmDe3x3.mtx";
        exp = new CMatrix(MatrixMarketMatrices.CmDe3x3);
        act = (CMatrix) mmParser.read(fileName);

        assertEquals(exp, act);

        // ---------------- sub-case 2 ----------------
        fileName = "./src/test/data/matrix_market/CmDe5x2.mtx";
        exp = new CMatrix(MatrixMarketMatrices.CmDe5x2);
        act = (CMatrix) mmParser.read(fileName);

        assertEquals(exp, act);
    }
}
