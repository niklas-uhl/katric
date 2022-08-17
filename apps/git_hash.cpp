#include "git_hash.h"

#ifdef CETRIC_REPO_GIT_HASH
    #include "git_version.h"
const std::string cetric::git_hash = kGitHash;
#else
    #ifdef CETRIC_FILE_GIT_HASH
        #include "commit.h"
const std::string cetric::git_hash = std::string(commit_data, commit_data + commit_size - 1);
    #else
const std::string cetric::git_hash = "undefined";
    #endif
#endif
