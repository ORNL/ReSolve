// this is for standalone testing (will NOT  be used in hiop)
//

class resolveMatrixIO{
  public:
    resolveMatrixIO();
    ~resolveMatrixIO();
    
    resolveMatrix* readMatrixFromFile(std::string filename);
    void* readAndUpdate(std::string filename, resolveMatrix* A);
    double* readRhsFromFile(std::string filename); 

  private:
    resolveMatrix* A;
    double* rhs;


};
